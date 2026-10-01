"""Text frontends for language-specific CtrlSpeech preprocessing."""

from .japanese import (
    JapaneseFrontend,
    JapaneseFrontendResult,
    JapaneseMorpheme,
)

__all__ = [
    "JapaneseFrontend",
    "JapaneseFrontendResult",
    "JapaneseMorpheme",
]
