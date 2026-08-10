"""Merkezi hata yönetimi.

Tüm API hataları tek bir formatta ve standart İngilizce kodlarla döner.
Kullanıcıya gösterilecek Türkçe mesajlar ayrı alanda bulunur.

Standart yanıt:
    {
        "error_code": "MODEL_UNAVAILABLE",
        "message": "Aktif model bulunamadı.",
        "details": {},
        "request_id": "uuid"
    }
"""

from __future__ import annotations

from typing import Any


class AppError(Exception):
    """Uygulama genelinde kullanılan temel hata sınıfı.

    Attributes:
        error_code: Sabit İngilizce hata kodu (örn. MODEL_UNAVAILABLE).
        message: Kullanıcıya gösterilecek Türkçe mesaj.
        status_code: HTTP durum kodu.
        details: Hatayla ilgili ek bağlam.
    """

    error_code: str = "INTERNAL_ERROR"
    message: str = "Beklenmeyen bir hata oluştu."
    status_code: int = 500
    details: dict[str, Any] | None = None

    def __init__(
        self,
        message: str | None = None,
        *,
        error_code: str | None = None,
        status_code: int | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        if message is not None:
            self.message = message
        if error_code is not None:
            self.error_code = error_code
        if status_code is not None:
            self.status_code = status_code
        self.details = details if details is not None else {}
        super().__init__(self.message)


class ModelUnavailableError(AppError):
    """Aktif model bulunamadığında fırlatılır."""

    error_code = "MODEL_UNAVAILABLE"
    message = "Aktif model bulunamadı. Tahmin servisi şu anda kullanılamıyor."
    status_code = 503


class ProcessNotFoundError(AppError):
    """İstenen süreç kaydı bulunamadığında fırlatılır."""

    error_code = "PROCESS_NOT_FOUND"
    message = "Süreç kaydı bulunamadı."
    status_code = 404


class SchemaMismatchError(AppError):
    """Girdi şeması modelin beklediği şema ile uyuşmadığında fırlatılır."""

    error_code = "SCHEMA_MISMATCH"
    message = "Girdi şeması modelin beklediği şema ile uyuşmuyor."
    status_code = 422


class InvalidSimulationFieldError(AppError):
    """Simülasyonda izin verilmeyen bir alan değiştirilmeye çalışıldığında fırlatılır."""

    error_code = "INVALID_SIMULATION_FIELD"
    message = "Bu alan simülasyon için izin verilen bir özellik değil."
    status_code = 422


class SnapshotNotFoundError(AppError):
    """Prediction için opening snapshot bulunamadığında fırlatılır."""

    error_code = "SNAPSHOT_NOT_FOUND"
    message = "Bu süreç için opening snapshot bulunamadı."
    status_code = 404


class ActiveModelAmbiguousError(AppError):
    """Birden fazla is_active=1 bundle bulunduğunda fırlatılır."""

    error_code = "ACTIVE_MODEL_AMBIGUOUS"
    message = "Birden fazla aktif model bundle bulundu. Deterministik seçim yapılamıyor."
    status_code = 503


class PredictionCompatibilityError(AppError):
    """Snapshot şeması veya mapping sürümü model bundle ile uyuşmadığında fırlatılır."""

    error_code = "PREDICTION_COMPATIBILITY_ERROR"
    message = "Snapshot ile model bundle uyumsuz."
    status_code = 422
