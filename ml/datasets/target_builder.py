"""Süreç outcome alanlarindan hedef degiskenleri hesaplama.

Hedef kaynagi: processes tablosu (completed_at, deadline, created_at)
Feature kaynagi degildir.
"""

from __future__ import annotations

from datetime import datetime

OBSERVATION_CUTOFF = datetime(2025, 1, 13, 0, 0, 0)


def compute_is_delayed(
    completed_at: datetime | None,
    deadline: datetime | None,
    observation_cutoff: datetime = OBSERVATION_CUTOFF,
) -> int | None:
    """Siniflandirma hedefini (is_delayed) hesaplar.

    Polika:
        A) completed_at dolu ve completed_at <= deadline → 0
        B) completed_at dolu ve completed_at > deadline → 1
        C) completed_at bos ve deadline < observation_cutoff → 1
        D) completed_at bos ve deadline >= observation_cutoff → None (dislanmis)
        E) deadline bos → None (dislanmis)

    Dönüs: 0, 1 veya None (dataset disi).
    """
    if deadline is None:
        return None

    if completed_at is not None:
        return 0 if completed_at <= deadline else 1

    if deadline < observation_cutoff:
        return 1

    return None


def compute_total_duration_hours(
    created_at: datetime,
    completed_at: datetime | None,
) -> float | None:
    """Regresyon hedefini (total_duration_hours) hesaplar.

    Kohort: created_at dolu, completed_at dolu, completed_at >= created_at.
    Açik süreçler veya negatif süreli kayitlar → None (dislanmis).
    """
    if completed_at is None:
        return None

    duration = (completed_at - created_at).total_seconds() / 3600

    if duration < 0:
        return None

    return float(duration)
