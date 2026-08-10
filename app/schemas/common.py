"""Ortak yanıt ve hata şemaları."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class ErrorResponse(BaseModel):
    """Standart hata yanıt formatı.

    error_code her zaman sabit İngilizce bir koddur.
    message kullanıcıya gösterilecek Türkçe açıklamadır.
    """

    error_code: str
    message: str
    details: dict[str, Any] = {}
    request_id: str
