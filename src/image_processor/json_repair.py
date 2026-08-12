"""
json_repair.py
==============
Parsing JSON yang toleran terhadap kesalahan kecil khas keluaran LLM.

MASALAH YANG DIPERBAIKI (B-45, 2026-08-11)
------------------------------------------
`qwen2.5vl:7b` menghasilkan JSON yang HAMPIR valid, tapi sering LUPA KOMA
persis di tempat ia menyisipkan baris kosong sebagai pemisah kelompok:

    "key_value_label_examples": ["Nama", "Pendidikan"]      <-- koma hilang
                                                            <-- baris kosong
    "has_data_table": false,
    ...
    "list_style": "sequential"                              <-- koma hilang

    "has_signature_block": true,

Akibatnya `json.loads` gagal dengan
    Expecting ',' delimiter: line 17 column 5
dan Phase 1 melempar RuntimeError -> SELURUH dokumen ditandai `failed`,
padahal isi analisisnya sendiri sudah benar dan lengkap.

Kejadian nyata: 3 dari 3 gambar tulisan tangan gagal berurutan karena ini.

Pendekatan yang dipakai:
  1. Coba `json.loads` apa adanya -- kalau sudah valid, tidak ada yang disentuh.
  2. Kalau gagal, tambal koma yang hilang lalu buang trailing comma, dan coba
     lagi. Perbaikannya konservatif: HANYA menyisipkan koma di antara akhir
     sebuah value dan awal key berikutnya. Tidak ada nilai yang diubah, jadi
     data hasil ekstraksi tidak mungkin terdistorsi.

Perbaikan ini sengaja dibuat provider-agnostic: Gemini pun kadang melakukan hal
serupa, dan biaya menjalankannya nol saat JSON-nya memang sudah valid.
"""

import json
import re


def _strip_fences(text: str) -> str:
    """Ambil objek JSON dari dalam prosa / code fence."""
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return match.group(0).strip() if match else text.strip()


def repair_json(text: str) -> str:
    """
    Tambal koma yang hilang di antara pasangan key/value.

    Menyisipkan koma ketika sebuah value berakhir dengan `"`, `]`, `}`, angka,
    `true`, `false`, atau `null`, lalu diikuti (setelah whitespace/newline)
    oleh sebuah key berupa string. Whitespace aslinya dipertahankan supaya
    nomor baris pada pesan error tetap masuk akal saat debugging.
    """
    # value berakhir dengan " ] } angka true false null  ->  spasi  ->  "key":
    pattern = re.compile(
        r'(?P<end>"|\]|\}|\d|\btrue\b|\bfalse\b|\bnull\b)'
        r'(?P<gap>\s*\n\s*)'
        r'(?P<next>"(?:[^"\\]|\\.)*"\s*:)'
    )
    repaired = pattern.sub(lambda m: f"{m.group('end')},{m.group('gap')}{m.group('next')}", text)

    # Buang trailing comma sebelum penutup, kalau langkah di atas kebablasan
    # atau model memang menuliskannya.
    repaired = re.sub(r",(\s*[}\]])", r"\1", repaired)
    return repaired


def loads_lenient(text: str) -> dict:
    """
    Parse JSON dari keluaran LLM. Melempar `json.JSONDecodeError` kalau tetap
    tidak bisa diselamatkan, supaya pemanggil bisa memberi konteksnya sendiri.
    """
    cleaned = _strip_fences(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    repaired = repair_json(cleaned)
    result = json.loads(repaired)  # kalau masih gagal, biarkan naik
    print("[JSON] Keluaran model tidak valid; koma yang hilang berhasil ditambal.")
    return result
