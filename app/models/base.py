"""SQLAlchemy declarative taban sınıfı."""

from __future__ import annotations

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Tüm ORM modellerinin türetildiği taban sınıf."""
