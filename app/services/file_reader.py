"""CSV ve XLSX dosya okuyucu.

Desteklenen formatlar:
    - .csv (pandas üzerinden)
    - .xlsx (openpyxl üzerinden)

Desteklenmeyen formatlar (.xls dahil) UNSUPPORTED_FILE_FORMAT hatasıyla reddedilir.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

SUPPORTED_EXTENSIONS = {".csv", ".xlsx"}


@dataclass(frozen=True)
class FileReadResult:
    """Dosya okuma sonucu."""

    rows: list[dict[str, object]]
    columns: list[str]
    row_count: int


class UnsupportedFileFormatError(Exception):
    """Desteklenmeyen dosya formatı."""

    def __init__(self, extension: str) -> None:
        self.extension = extension
        super().__init__(
            f"Desteklenmeyen dosya formati: {extension}. "
            f"Desteklenen formatlar: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )


class FileReadError(Exception):
    """Dosya okunamadı (bozuk veya erişilemez)."""

    pass


def read_data_file(file_path: str | Path) -> FileReadResult:
    """CSV veya XLSX dosyasını okur ve satır listesi döndürür.

    Args:
        file_path: Dosya yolu.

    Returns:
        FileReadResult: Satırlar, kolonlar ve satır sayısı.

    Raises:
        UnsupportedFileFormatError: Format desteklenmiyorsa.
        FileReadError: Dosya okunamıyorsa.
    """
    path = Path(file_path)
    extension = path.suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFileFormatError(extension)

    try:
        if extension == ".csv":
            df = pd.read_csv(path, dtype=str, keep_default_na=False)
        else:
            df = pd.read_excel(path, dtype=str, keep_default_na=False, engine="openpyxl")
    except pd.errors.EmptyDataError as exc:
        raise FileReadError("Dosya bos veya kolon icermiyor.") from exc
    except pd.errors.ParserError as exc:
        raise FileReadError("Dosya parse edilemedi.") from exc
    except Exception as exc:
        raise FileReadError(f"Dosya okunamadi: {type(exc).__name__}") from exc

    rows = df.to_dict(orient="records")
    columns = df.columns.tolist()

    return FileReadResult(rows=rows, columns=columns, row_count=len(rows))
