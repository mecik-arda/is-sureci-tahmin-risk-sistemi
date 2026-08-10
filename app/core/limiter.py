"""Rate limiting yapilandirmasi (slowapi).

Bu modul ana uygulama ve route'lar arasinda paylasilir.
"""

from __future__ import annotations

from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address, default_limits=["120/minute"])
