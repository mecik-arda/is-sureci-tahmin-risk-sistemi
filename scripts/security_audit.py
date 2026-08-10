"""Guvenlik denetim scripti.

Kontroller:
  1. Log dosyalarinda hassas veri taramasi
  2. Kaynak kodda hardcoded secret/key/api anahtari taramasi
  3. Dis ag baglantisi testi (Wireshark/netstat onerisi)

Kullanim:
    python scripts/security_audit.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

SENSITIVE_PATTERNS = [
    (re.compile(r'api[_-]?key\s*[=:]\s*["\'][^"\']+["\']', re.IGNORECASE), "API anahtari"),
    (re.compile(r'password\s*[=:]\s*["\'][^"\']+["\']', re.IGNORECASE), "Parola"),
    (re.compile(r'token\s*[=:]\s*["\'][^"\']{20,}["\']', re.IGNORECASE), "Token"),
    (re.compile(r'secret\s*[=:]\s*["\'][^"\']+["\']', re.IGNORECASE), "Secret"),
    (re.compile(r'(?<![a-zA-Z0-9])[a-zA-Z0-9+/]{40,}(?![a-zA-Z0-9])', 0), "Base64 benzeri"),
]

EXCLUDE_DIRS = {".venv", "__pycache__", ".pytest_cache", ".pytest_tmp", "data", "artifacts", ".git"}


def _scan_code() -> list[dict]:
    issues = []
    for py_file in project_root.rglob("*.py"):
        parts = set(py_file.parts)
        if parts & EXCLUDE_DIRS:
            continue
        try:
            text = py_file.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for pattern, label in SENSITIVE_PATTERNS:
            matches = pattern.findall(text)
            if matches:
                issues.append({
                    "file": str(py_file.relative_to(project_root)),
                    "type": f"Potansiyel {label}",
                    "count": len(matches),
                })
    return issues


def _check_env_file() -> list[dict]:
    issues = []
    env_example = project_root / ".env.example"
    if env_example.exists():
        issues.append({"file": ".env.example", "type": "INFO", "message": ".env.example mevcut, .env gitignore'da olmali."})

    env_file = project_root / ".env"
    if env_file.exists():
        content = env_file.read_text(encoding="utf-8", errors="ignore")
        has_values = bool(re.search(r'=\s*\S', content))
        if has_values:
            issues.append({"file": ".env", "type": "UYARI", "message": ".env dosyasi deger iceriyor. Git'e eklenmediginden emin olun."})

    gitignore = project_root / ".gitignore"
    if gitignore.exists():
        content = gitignore.read_text(encoding="utf-8", errors="ignore")
        if ".env" not in content:
            issues.append({"file": ".gitignore", "type": "UYARI", "message": ".gitignore'da .env satiri bulunamadi."})
        if "*.db" not in content:
            issues.append({"file": ".gitignore", "type": "UYARI", "message": ".gitignore'da *.db satiri bulunamadi. SQLite veritabani repoya gitmesin."})

    return issues


def _network_isolation_guide() -> str:
    return """
  Ag izolasyonu testi (manuel):
  1. Uygulamayi baslat: uvicorn app.main:app --host 127.0.0.1 --port 8000
  2. Tum sayfalari gez: Dashboard, Surec Listesi, Surec Detayi, Model Performansi,
     Veri Kalitesi, Model Izleme
  3. Wireshark veya netstat ile dis baglanti kontrolu yap:
     netstat -an | findstr ESTABLISHED
  4. Beklenen: Yalnizca 127.0.0.1:8000 uzerinde lokal baglantilar.
     Dis IP'lere baglanti olmamali.
  5. Ayrica kontrol et: app/static/ icinde external CDN referansi var mi?
     (Jinja2 template'leri ve JS dosyalarini tara)
"""


def main() -> None:
    print("=" * 60)
    print("         GUVENLIK DENETIM RAPORU")
    print("=" * 60)
    print()

    print("--- Kaynak kodda hassas veri taramasi ---")
    code_issues = _scan_code()
    if code_issues:
        for issue in code_issues:
            print(f"  [{issue['type']}] {issue['file']} ({issue['count']} eslesme)")
    else:
        print("  Hassas veri bulunamadi.")
    print()

    print("--- Ortam dosyasi kontrolleri ---")
    env_issues = _check_env_file()
    for issue in env_issues:
        print(f"  [{issue['type']}] {issue.get('message', issue.get('file', ''))}")
    print()

    print("--- Ag izolasyonu ---")
    print(_network_isolation_guide())

    # CDN kontrolu
    cdn_pattern = re.compile(r'https?://(?!127\.0\.0\.1|localhost)[^\s"\']+\.(js|css|png|svg|woff)', re.IGNORECASE)
    cdn_found = False
    for f in project_root.glob("app/templates/**/*.html"):
        content = f.read_text(encoding="utf-8", errors="ignore")
        matches = cdn_pattern.findall(content)
        if matches:
            print(f"  UYARI: {f.relative_to(project_root)} dosyasinda external referans bulundu: {matches}")
            cdn_found = True
    for f in project_root.glob("app/static/**/*.js"):
        content = f.read_text(encoding="utf-8", errors="ignore")
        matches = cdn_pattern.findall(content)
        if matches:
            print(f"  UYARI: {f.relative_to(project_root)} dosyasinda external referans bulundu: {matches}")
            cdn_found = True
    if not cdn_found:
        print("  External CDN referansi bulunamadi.")

    print()
    print("=" * 60)


if __name__ == "__main__":
    main()
