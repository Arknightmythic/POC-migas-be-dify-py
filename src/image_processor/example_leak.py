"""
example_leak.py
===============
Deteksi kebocoran isi contoh few-shot ke dalam hasil OCR.

MASALAH YANG DIPERBAIKI (B-47, 2026-08-11)
------------------------------------------
Prompt Phase 3 memuat EXAMPLE snippet HTML sebagai cetakan struktur. Contoh-
contoh itu memakai teks dari sebuah surat dinas nyata (Tembusan ke Presiden RI,
BADAN INTELIJEN NEGARA, "( ERRY MARSONO )", "Ditetapkan di : Jakarta",
"16 Januari 2017").

qwen2.5vl:7b menyalin teks itu apa adanya ke dalam hasil OCR sebuah FORMULIR
SIDANG SKRIPSI kampus -- dokumen yang isinya sama sekali berbeda. Hasilnya:
file .docx yang tampak resmi tapi memuat blok yang DIKARANG dan tidak ada di
gambar aslinya. Ini jauh lebih berbahaya daripada sekadar OCR meleset, karena
kesalahannya tidak terlihat seperti kesalahan.

Perbaikan berlapis:
  1. Instruksi prompt dipertegas: contoh adalah STRUKTUR saja (lihat B-47 di
     build_extraction_prompt).
  2. Modul ini: kalau instruksi tetap diabaikan, kebocorannya terdeteksi dan
     `_validate_html` memicu retry Phase 3.

Frasa di bawah dipilih yang sangat spesifik terhadap contoh, sehingga hampir
mustahil muncul secara sah pada dokumen KAI/kampus. Kalau suatu saat memang ada
dokumen yang sah memuat salah satunya, hapus frasa itu dari daftar.
"""

# Frasa khas dari EXAMPLE snippet di doc_ocr_pipeline.py & handwritten_pipeline.py
EXAMPLE_PHRASES = [
    "erry marsono",
    "badan intelijen negara",
    "menkopolhukam",
    "panglima tni",
    "presiden republik indonesia",
    "satkorlak",
    "16 januari 2017",
    "opsinsus",
]


def find_example_leaks(html: str) -> list:
    """Kembalikan daftar frasa contoh yang bocor ke dalam `html`."""
    low = (html or "").lower()
    return [p for p in EXAMPLE_PHRASES if p in low]
