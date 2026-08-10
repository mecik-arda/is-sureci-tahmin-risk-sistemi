"""Sealed holdout uygulamasi.

Faz 3'te test/audit degerlendirmesini engeller.
Faz 5'te final degerlendirme icin muhur acilabilir.
"""

from __future__ import annotations

from typing import Final

ALLOWED_EVAL_SPLITS: Final[frozenset[str]] = frozenset({"train", "validation"})
SEALED_SPLITS: Final[frozenset[str]] = frozenset({"test", "audit"})
ALL_SPLITS: Final[frozenset[str]] = ALLOWED_EVAL_SPLITS | SEALED_SPLITS

_phase5_unsealed: bool = False


class SealedSplitError(Exception):
    """Sealed bir split uzerinde evaluation yapilmaya calisildiginda firlatilir."""


def unseal_for_phase5() -> None:
    global _phase5_unsealed
    _phase5_unsealed = True


def is_unsealed() -> bool:
    return _phase5_unsealed


def assert_evaluable(split_name: str) -> None:
    allowed = ALL_SPLITS if _phase5_unsealed else ALLOWED_EVAL_SPLITS
    if split_name not in ALL_SPLITS:
        raise SealedSplitError(
            f"Bilinmeyen split: '{split_name}'. "
            f"Gecerli split'ler: {sorted(ALL_SPLITS)}"
        )
    if split_name not in allowed:
        raise SealedSplitError(
            f"Split '{split_name}' muhurludur. "
            f"Izin verilen split'ler: {sorted(allowed)}"
        )
