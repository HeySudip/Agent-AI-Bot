"""SSRF protection for outbound URL fetches.

Validates a URL before any HTTP request leaves the process. Refuses non-HTTP
schemes and blocks resolution to private, loopback, link-local, multicast,
or reserved IP ranges. This prevents the agent's URL-fetching tools from
being used to reach internal services (cloud metadata endpoints,
intranet hosts, etc.).
"""

from __future__ import annotations

import ipaddress
import socket
from typing import Final
from urllib.parse import urlparse

__all__ = ["SSRFBlockedError", "assert_url_is_safe", "is_url_safe"]


class SSRFBlockedError(ValueError):
    """Raised when a URL targets a forbidden network or scheme."""


_ALLOWED_SCHEMES: Final[frozenset[str]] = frozenset({"http", "https"})
_BLOCKED_HOSTNAMES: Final[frozenset[str]] = frozenset(
    {
        "localhost",
        "ip6-localhost",
        "ip6-loopback",
        "broadcasthost",
        # Common cloud metadata service hostnames
        "metadata.google.internal",
        "metadata",
        "instance-data",
    }
)


def is_url_safe(url: str) -> tuple[bool, str | None]:
    """Return ``(safe, reason)``. ``reason`` is None when safe."""
    try:
        assert_url_is_safe(url)
    except SSRFBlockedError as exc:
        return False, str(exc)
    return True, None


def assert_url_is_safe(url: str) -> None:
    """Validate ``url``. Raise :class:`SSRFBlockedError` on any policy failure.

    Performs DNS resolution and rejects URLs that resolve to disallowed
    address ranges. DNS resolution may block; callers should run this in a
    thread when used from async code, or wrap with a timeout.
    """
    if not isinstance(url, str) or not url:
        raise SSRFBlockedError("URL must be a non-empty string.")

    parsed = urlparse(url)
    scheme = (parsed.scheme or "").lower()
    if scheme not in _ALLOWED_SCHEMES:
        raise SSRFBlockedError(
            f"URL scheme {scheme!r} is not allowed (only http/https)."
        )

    hostname = (parsed.hostname or "").lower()
    if not hostname:
        raise SSRFBlockedError("URL has no hostname.")

    if hostname in _BLOCKED_HOSTNAMES:
        raise SSRFBlockedError(f"Hostname {hostname!r} is blocked.")

    # AWS-style metadata endpoint check on the hostname itself, before DNS.
    if hostname == "169.254.169.254":
        raise SSRFBlockedError("Cloud metadata endpoint is blocked.")

    try:
        infos = socket.getaddrinfo(hostname, None)
    except socket.gaierror as exc:
        raise SSRFBlockedError(f"DNS resolution failed for {hostname!r}: {exc}") from exc

    if not infos:
        raise SSRFBlockedError(f"No addresses resolved for {hostname!r}.")

    for info in infos:
        sockaddr = info[4]
        ip_str = sockaddr[0]
        try:
            ip = ipaddress.ip_address(ip_str)
        except ValueError as exc:
            raise SSRFBlockedError(f"Could not parse address {ip_str!r}.") from exc
        _assert_ip_is_safe(ip, hostname)


def _assert_ip_is_safe(
    ip: ipaddress.IPv4Address | ipaddress.IPv6Address, hostname: str
) -> None:
    if ip.is_loopback:
        raise SSRFBlockedError(f"{hostname!r} resolves to loopback ({ip}).")
    if ip.is_private:
        raise SSRFBlockedError(f"{hostname!r} resolves to a private network ({ip}).")
    if ip.is_link_local:
        raise SSRFBlockedError(f"{hostname!r} resolves to link-local ({ip}).")
    if ip.is_multicast:
        raise SSRFBlockedError(f"{hostname!r} resolves to multicast ({ip}).")
    if ip.is_reserved:
        raise SSRFBlockedError(f"{hostname!r} resolves to a reserved range ({ip}).")
    if ip.is_unspecified:
        raise SSRFBlockedError(f"{hostname!r} resolves to unspecified address.")
