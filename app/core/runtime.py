"""Uygulama çalışma zamanı durumu.

Lifespan başlangıcında model yüklenirse durum 'ready' olur.
Model bulunamazsa uygulama 'degraded' modda çalışmaya devam eder;
yalnızca model gerektiren endpoint'ler MODEL_UNAVAILABLE döndürür.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RuntimeState:
    """Uygulama geneli paylaşılan durum.

    NOT: Bu nesne HTTP istekleri arasında durum paylaşmak için kullanılır,
    fakat hiçbir oturum veya kullanıcı verisi içermez.
    """

    model_available: bool = False
    active_model_version: str | None = None
    startup_completed: bool = False
    startup_errors: list[str] = field(default_factory=list)
    bundle: Any = None

    def mark_ready(self, model_version: str, bundle: Any) -> None:
        self.model_available = True
        self.active_model_version = model_version
        self.bundle = bundle

    def mark_degraded(self, reason: str) -> None:
        self.model_available = False
        self.active_model_version = None
        self.bundle = None
        self.startup_errors.append(reason)

    def mark_startup_done(self) -> None:
        self.startup_completed = True


# Tekil örnek
runtime_state = RuntimeState()
