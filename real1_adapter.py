"""Wrapper for integrating Real1 AGI capabilities.

This module attempts to dynamically load the
`Real1_AGI_Monolithic_MERGED.py` script and expose a small
interface that can be used by other components.

Because the original Real1 script is extremely large and may contain
syntax issues depending on the source, loading is performed lazily
and failures are surfaced with clear error messages.

The primary function provided is `run(prompt: str) -> str` which will
create an instance of the `AutonomousAgent` class defined in the Real1
script and attempt to produce a response for the given prompt.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Optional

_REAL1_MODULE = None
_IMPORT_ERROR: Optional[Exception] = None


def _load_real1_module():
    """Attempt to import the Real1 monolithic script.

    Returns the imported module on success or ``None`` on failure. Any import
    failure is stored in the global ``_IMPORT_ERROR`` variable for diagnostics.
    """
    global _REAL1_MODULE, _IMPORT_ERROR
    if _REAL1_MODULE is not None or _IMPORT_ERROR is not None:
        return _REAL1_MODULE

    path = Path(__file__).with_name("Real1_AGI_Monolithic_MERGED.py")
    if not path.exists():
        _IMPORT_ERROR = FileNotFoundError(f"Real1 script not found at {path}")
        return None

    try:
        spec = importlib.util.spec_from_file_location("real1_impl", path)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)  # type: ignore[arg-type]
            _REAL1_MODULE = module
        else:
            _IMPORT_ERROR = ImportError("spec loader missing for Real1 module")
    except Exception as exc:  # pragma: no cover - diagnostics only
        _IMPORT_ERROR = exc
    return _REAL1_MODULE


def capabilities_available() -> bool:
    """Return ``True`` if the Real1 capabilities are importable."""
    return _load_real1_module() is not None


def run(prompt: str) -> str:
    """Execute the Real1 agent on the provided ``prompt``.

    Raises ``RuntimeError`` if the Real1 module or a suitable interface could
    not be located. This keeps the rest of the system safe from partial
    integrations.
    """
    mod = _load_real1_module()
    if mod is None:
        raise RuntimeError(f"Real1 capabilities unavailable: {_IMPORT_ERROR}")

    agent_cls = getattr(mod, "AutonomousAgent", None)
    if agent_cls is None:
        raise RuntimeError("AutonomousAgent class not found in Real1 module")

    agent = agent_cls()

    # Prefer a dedicated run method if exposed.
    if hasattr(agent, "run"):
        return agent.run(prompt)  # type: ignore[return-value]

    # Fall back to an LLM wrapper chat method if available.
    wrapper = getattr(agent, "llm_wrapper", None)
    if wrapper and hasattr(wrapper, "chat"):
        return wrapper.chat(prompt)

    raise RuntimeError("No callable interface found for Real1 agent")
