"""
console_utf8.py
===============
Paksa stdout/stderr memakai UTF-8.

MASALAH YANG DIPERBAIKI (B-28, 2026-08-10)
------------------------------------------
Kode di project ini banyak memakai emoji dan panah pada print(), misalnya:

    print("🚀 Uploading ...")
    print("[DOCX] Converting HTML → result.docx ...")

Di Windows, saat stdout diarahkan ke FILE atau PIPE (bukan console), Python
memakai encoding locale -- di mesin ini cp1252 -- yang tidak bisa meng-encode
karakter tersebut, sehingga print() melempar UnicodeEncodeError dan
menghentikan proses di tengah jalan:

    UnicodeEncodeError: 'charmap' codec can't encode character '\\u2192'

Efeknya tidak kelihatan saat dijalankan langsung di terminal (Windows console
ditulisi sebagai UTF-16, jadi aman), tapi langsung muncul begitu output
di-redirect. Padahal itu justru cara menjalankan yang dianjurkan README:

    nohup celery -A ... worker --loglevel=info > celery_worker.log 2>&1 &
    docker compose up -d            # log container juga di-capture

Konkretnya: konversi HTML -> DOCX di pipeline OCR gagal karena baris print
berisi karakter panah, bukan karena logika konversinya.

errors="replace" dipakai sebagai pengaman terakhir supaya terminal dengan
code page aneh pun tidak pernah bisa membuat proses mati hanya gara-gara
karakter yang tidak bisa ditampilkan.
"""

import sys


def enable_utf8_console() -> None:
    """Set stdout & stderr ke UTF-8. Aman dipanggil berkali-kali."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            # AttributeError: stream sudah diganti objek lain (mis. saat test).
            # ValueError  : stream sudah ditutup.
            pass
