"""Tests for the SSRF guard.

We monkeypatch ``socket.getaddrinfo`` so the tests are deterministic and do
not depend on real DNS or network access.
"""

from __future__ import annotations

import socket
from typing import Any

import pytest

from safety import ssrf_guard
from safety.ssrf_guard import SSRFBlockedError, assert_url_is_safe, is_url_safe


def _fake_resolver(addresses: list[str]) -> Any:
    def _resolve(_host: str, _port: Any) -> list[Any]:
        out: list[Any] = []
        for addr in addresses:
            family = socket.AF_INET6 if ":" in addr else socket.AF_INET
            out.append((family, socket.SOCK_STREAM, 0, "", (addr, 0)))
        return out

    return _resolve


class TestSchemes:
    def test_http_scheme_allowed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(ssrf_guard.socket, "getaddrinfo", _fake_resolver(["93.184.216.34"]))
        assert_url_is_safe("http://example.com/")

    def test_https_scheme_allowed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(ssrf_guard.socket, "getaddrinfo", _fake_resolver(["93.184.216.34"]))
        assert_url_is_safe("https://example.com/")

    @pytest.mark.parametrize("scheme", ["file", "ftp", "gopher", "javascript", "data"])
    def test_other_schemes_blocked(self, scheme: str) -> None:
        with pytest.raises(SSRFBlockedError):
            assert_url_is_safe(f"{scheme}://example.com/")


class TestBlockedHostnames:
    def test_localhost_blocked(self) -> None:
        with pytest.raises(SSRFBlockedError):
            assert_url_is_safe("http://localhost/")

    def test_aws_metadata_ip_blocked(self) -> None:
        with pytest.raises(SSRFBlockedError):
            assert_url_is_safe("http://169.254.169.254/latest/meta-data/")

    def test_gcp_metadata_hostname_blocked(self) -> None:
        with pytest.raises(SSRFBlockedError):
            assert_url_is_safe("http://metadata.google.internal/")


class TestIPRanges:
    @pytest.mark.parametrize(
        "ip",
        [
            "127.0.0.1",
            "10.0.0.1",
            "172.16.0.1",
            "192.168.1.1",
            "169.254.1.1",
            "0.0.0.0",
            "224.0.0.1",
        ],
    )
    def test_blocked_ip_ranges(self, ip: str, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(ssrf_guard.socket, "getaddrinfo", _fake_resolver([ip]))
        with pytest.raises(SSRFBlockedError):
            assert_url_is_safe("http://example.com/")

    def test_blocked_ipv6_loopback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(ssrf_guard.socket, "getaddrinfo", _fake_resolver(["::1"]))
        with pytest.raises(SSRFBlockedError):
            assert_url_is_safe("http://example.com/")

    def test_blocked_unique_local_ipv6(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(ssrf_guard.socket, "getaddrinfo", _fake_resolver(["fc00::1"]))
        with pytest.raises(SSRFBlockedError):
            assert_url_is_safe("http://example.com/")

    def test_public_ip_allowed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(ssrf_guard.socket, "getaddrinfo", _fake_resolver(["93.184.216.34"]))
        assert_url_is_safe("https://example.com/path")

    def test_dns_failure_is_blocked(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def boom(*_a: Any, **_kw: Any) -> Any:
            raise socket.gaierror("no such host")

        monkeypatch.setattr(ssrf_guard.socket, "getaddrinfo", boom)
        with pytest.raises(SSRFBlockedError):
            assert_url_is_safe("https://does-not-exist.example/")


class TestPublicAPI:
    def test_is_url_safe_returns_tuple(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(ssrf_guard.socket, "getaddrinfo", _fake_resolver(["127.0.0.1"]))
        ok, reason = is_url_safe("http://example.com/")
        assert ok is False
        assert reason is not None and "loopback" in reason

    def test_empty_url_blocked(self) -> None:
        with pytest.raises(SSRFBlockedError):
            assert_url_is_safe("")
