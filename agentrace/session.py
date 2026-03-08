"""Session and user tracking via context variables."""

from __future__ import annotations

from contextvars import ContextVar, Token
from contextlib import contextmanager
from typing import Generator

_session_id_var: ContextVar[str | None] = ContextVar("agentrace_session_id", default=None)
_user_id_var: ContextVar[str | None] = ContextVar("agentrace_user_id", default=None)


def set_session(session_id: str) -> Token[str | None]:
    """Set the session ID for all subsequent spans in this context."""
    return _session_id_var.set(session_id)


def set_user(user_id: str) -> Token[str | None]:
    """Set the user ID for all subsequent spans in this context."""
    return _user_id_var.set(user_id)


def get_session() -> str | None:
    return _session_id_var.get()


def get_user() -> str | None:
    return _user_id_var.get()


@contextmanager
def session_context(
    session_id: str | None = None, user_id: str | None = None
) -> Generator[None, None, None]:
    """Scoped session/user context. Restores previous values on exit."""
    tokens: list[Token] = []
    if session_id is not None:
        tokens.append(_session_id_var.set(session_id))
    if user_id is not None:
        tokens.append(_user_id_var.set(user_id))
    try:
        yield
    finally:
        for tok in reversed(tokens):
            if session_id is not None and tok == tokens[0]:
                _session_id_var.reset(tok)
            else:
                _user_id_var.reset(tok)
