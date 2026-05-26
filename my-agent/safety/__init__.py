"""Safety primitives: safe expression evaluation, SSRF guard, secret redaction.

This package isolates security-sensitive code paths so they can be reviewed,
tested, and audited independently of the rest of the agent.
"""

from .safe_eval import SafeEvalError, safe_eval
from .secrets_redactor import redact_secrets
from .ssrf_guard import SSRFBlockedError, assert_url_is_safe

__all__ = [
    "SafeEvalError",
    "SSRFBlockedError",
    "assert_url_is_safe",
    "redact_secrets",
    "safe_eval",
]
