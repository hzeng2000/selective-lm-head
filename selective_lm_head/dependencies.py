"""Small helpers for optional runtime dependencies."""

from __future__ import annotations

import importlib
from types import ModuleType


def require_module(name: str, install_hint: str | None = None) -> ModuleType:
    """Import an optional dependency or raise an actionable ImportError."""

    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as exc:
        top_level = name.split(".", 1)[0]
        if exc.name != top_level:
            raise
        hint = install_hint or f"Install the optional dependency that provides {name!r}."
        raise ImportError(f"Missing optional dependency: {name}. {hint}") from exc


def module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None

