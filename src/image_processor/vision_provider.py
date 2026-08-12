"""
vision_provider.py
==================
Lapisan provider tunggal untuk semua panggilan vision-LLM di pipeline OCR
dokumen (doc_ocr_pipeline.py dan handwritten_pipeline.py).

LATAR BELAKANG (B-06 / B-25, 2026-08-10)
----------------------------------------
Sebelumnya kedua pipeline meng-hardcode:

    OLLAMA_URL = "http://103.67.43.152/ollama2/api/chat"
    MODEL      = "qwen3-vl:8b-instruct-bf16"

Dua masalah:
  1. Nilai .env (OLLAMA_URL / OLLAMA_VISION_MODEL) tidak pernah dibaca, jadi
     mengubah konfigurasi tidak berpengaruh apa pun.
  2. Host 103.67.43.152 tidak terjangkau (timeout), sehingga setiap dokumen
     kategori "administrative" menggantung ~300 detik lalu gagal.

Modul ini memindahkan pilihan provider ke environment dan menambahkan jalur
Gemini, supaya pipeline tetap jalan selama Ollama belum tersedia.

ENVIRONMENT
-----------
  VISION_PROVIDER       'gemini' (default) atau 'ollama'
  VISION_TIMEOUT        timeout detik per panggilan (default 300)

  # kalau VISION_PROVIDER=ollama
  OLLAMA_URL            base URL ATAU endpoint lengkap; dua-duanya diterima:
                          http://localhost:11434
                          http://localhost:11434/api/chat
  OLLAMA_VISION_MODEL   default 'qwen3-vl:8b-instruct-bf16'

  # kalau VISION_PROVIDER=gemini
  GEMINI_API_KEY        wajib
  GEMINI_VISION_MODEL   default mengikuti GEMINI_MODEL, lalu 'gemini-2.5-flash'

FORMAT PESAN
------------
Tetap memakai format ala Ollama supaya call site di kedua pipeline tidak perlu
berubah:

    [{"role": "system", "content": "..."},                      # opsional
     {"role": "user",   "content": "...", "images": [b64, ...]}]

Untuk Gemini, pesan ini diterjemahkan: role 'system' menjadi system_instruction,
teks user menjadi part teks, dan tiap gambar base64 menjadi inline_data.
"""

import base64
import binascii
import os
import re
import time

import requests

DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "qwen3-vl:8b-instruct-bf16"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_TIMEOUT = 300

# [B-50] Ollama Cloud memakai protokol yang SAMA persis dengan Ollama lokal
# (POST /api/chat, gambar sebagai images:[base64]), hanya beda host + perlu
# header Authorization. Jadi ini bukan provider baru dari nol, melainkan varian
# dari jalur Ollama yang sudah ada.
DEFAULT_OLLAMA_CLOUD_URL = "https://ollama.com"
DEFAULT_OLLAMA_CLOUD_MODEL = "gemma4:31b"


class VisionTruncated(RuntimeError):
    """
    [B-48] Provider memotong output karena kehabisan token.

    Dibedakan dari error lain supaya pemanggil (Phase 3) bisa memperlakukannya
    sebagai "coba lagi", bukan "dokumen rusak". Output yang terpotong terlihat
    seperti HTML yang sah, jadi kalau tidak dijadikan error ia akan lolos diam-
    diam dan menghasilkan dokumen yang tidak lengkap.
    """


# Akumulasi biaya selama proses berjalan, untuk log (bukan tagihan resmi).
_COST = {"calls": 0, "usd": 0.0}


def _price(env_name: str, default: float) -> float:
    """Harga per 1 token, dari harga per 1 juta token di env."""
    try:
        return float(os.getenv(env_name, default)) / 1_000_000
    except (TypeError, ValueError):
        return float(default) / 1_000_000


# ──────────────────────────────────────────────────────────────
# Helper
# ──────────────────────────────────────────────────────────────

def _provider() -> str:
    return os.getenv("VISION_PROVIDER", "gemini").strip().lower()


def _timeout() -> int:
    try:
        return int(os.getenv("VISION_TIMEOUT", DEFAULT_TIMEOUT))
    except ValueError:
        return DEFAULT_TIMEOUT


def _ollama_chat_url(cloud: bool = False) -> str:
    """Terima base URL maupun endpoint lengkap, kembalikan endpoint /api/chat."""
    env = "OLLAMA_CLOUD_URL" if cloud else "OLLAMA_URL"
    default = DEFAULT_OLLAMA_CLOUD_URL if cloud else DEFAULT_OLLAMA_URL
    raw = os.getenv(env, default).strip().rstrip("/")
    if raw.endswith("/api/chat"):
        return raw
    return f"{raw}/api/chat"


def _ollama_model(cloud: bool = False) -> str:
    if cloud:
        return os.getenv("OLLAMA_CLOUD_VISION_MODEL", DEFAULT_OLLAMA_CLOUD_MODEL)
    return os.getenv("OLLAMA_VISION_MODEL", DEFAULT_OLLAMA_MODEL)


def _sniff_mime(raw: bytes) -> str:
    """Tentukan mime type dari magic bytes. Gemini menolak mime yang salah."""
    if raw.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if raw.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return "image/webp"
    if raw.startswith(b"GIF87a") or raw.startswith(b"GIF89a"):
        return "image/gif"
    if raw.startswith(b"BM"):
        return "image/bmp"
    return "image/png"


def describe_provider(provider: str = None) -> str:
    """String satu baris untuk log startup / diagnostik."""
    p = (provider or _provider()).strip().lower()
    if p in ("ollama", "ollama-local"):
        return f"ollama ({_ollama_model(False)} @ {_ollama_chat_url(False)})"
    if p in ("ollama-cloud", "ollama_cloud"):
        key = os.getenv("OLLAMA_CLOUD_API_KEY", "")
        return (f"ollama-cloud ({_ollama_model(True)} @ {_ollama_chat_url(True)}, "
                f"key={'ada' if key else 'TIDAK ADA'})")
    return f"gemini ({_gemini_model()})"


def _gemini_model() -> str:
    return (
        os.getenv("GEMINI_VISION_MODEL")
        or os.getenv("GEMINI_MODEL")
        or DEFAULT_GEMINI_MODEL
    )


# ──────────────────────────────────────────────────────────────
# Ollama
# ──────────────────────────────────────────────────────────────

def _chat_ollama(messages: list, max_tokens: int, temperature: float,
                 json_mode: bool = False, cloud: bool = False) -> str:
    url = _ollama_chat_url(cloud)
    headers = {}

    if cloud:
        key = os.getenv("OLLAMA_CLOUD_API_KEY", "").strip()
        if not key:
            raise RuntimeError(
                "OLLAMA_CLOUD_API_KEY tidak di-set, padahal provider "
                "'ollama-cloud' dipilih."
            )
        headers["Authorization"] = f"Bearer {key}"

    # [B-46] num_ctx WAJIB di-set eksplisit.
    #
    # qwen2.5vl:7b mendukung context 128.000, tapi modelfile-nya hanya men-set
    # temperature -- jadi Ollama memakai DEFAULT-nya sendiri (4096). Gambar
    # dokumen memakan ribuan token vision, ditambah prompt Phase 3 (~1.300
    # token) dan output HTML yang panjang, sehingga 4096 langsung penuh dan
    # Ollama MEMOTONG output di tengah -- tanpa error, tanpa peringatan.
    #
    # Hasil pengukuran pada 4.jpeg (324 KB base64, form padat + tulisan tangan):
    #
    #     num_ctx   char HTML   teks   isi penting tertangkap
    #     4096(def)      1.194    314   1/8   <- HTML terpotong di tengah tag
    #     8192           5.836  1.158   3/8
    #     16384         11.520  1.848   7/8   <- dipakai sebagai default
    #     32768         10.554  1.598   6/8   <- tidak lebih baik, KV cache lebih berat
    #
    # Ini menjelaskan kenapa gambar KECIL berhasil sementara gambar BESAR gagal:
    # bukan soal kualitas model, murni context habis.
    #
    # Catatan VRAM: KV cache tumbuh seiring num_ctx. GPU di server RTX 2080 Ti
    # (11 GB) dan model Q4_K_M ~6 GB, jadi 16384 masih aman. Naikkan lewat
    # OLLAMA_NUM_CTX kalau pindah ke GPU yang lebih besar.
    try:
        num_ctx = int(os.getenv("OLLAMA_NUM_CTX", "16384"))
    except ValueError:
        num_ctx = 16384

    options = {
        "temperature": temperature,
        "top_p": 0.9,
        "num_predict": max_tokens,
    }

    if cloud:
        # [B-50] JANGAN kirim num_ctx ke Ollama Cloud.
        #
        # num_ctx=16384 itu perbaikan untuk server KANTOR, yang default-nya
        # cuma 4096 (B-46). Model cloud punya context JAUH lebih besar
        # (gemma4:31b = 262.144), jadi mengirim 16384 ke sana justru
        # MENURUNKAN-nya 16x -- mengulang bug B-46 dalam bentuk terbalik,
        # dan gejalanya persis sama: output terpotong diam-diam.
        # Biarkan cloud memakai default modelnya sendiri.
        #
        # think:false -- gemma4:31b punya kapabilitas "thinking". Pelajaran
        # B-48 berlaku sama persis: untuk transkripsi OCR, thinking hanya
        # menambah latensi & token tanpa memperbaiki hasil.
        payload = {
            "model": _ollama_model(cloud=True),
            "messages": messages,
            "stream": False,
            "think": os.getenv("OLLAMA_CLOUD_THINK", "false").strip().lower()
                     in ("1", "true", "yes"),
            "options": options,
        }
    else:
        options["num_ctx"] = num_ctx
        payload = {
            "model": _ollama_model(cloud=False),
            "messages": messages,
            "stream": False,
            "options": options,
        }
    # [B-45] Kunci output ke JSON valid di sisi server. qwen2.5vl kalau
    # dibiarkan bebas sering lupa koma di antara kelompok field, yang bikin
    # json.loads gagal dan seluruh dokumen ditandai failed.
    if json_mode:
        payload["format"] = "json"
    label = "Ollama Cloud" if cloud else "Ollama"
    t0 = time.time()
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=_timeout())
    except requests.RequestException as e:
        raise RuntimeError(
            f"Tidak bisa menghubungi {label} di {url}. "
            f"Cek {'OLLAMA_CLOUD_URL / API key' if cloud else 'OLLAMA_URL / apakah service-nya jalan'}. "
            f"Detail: {e!r}"
        ) from e

    # [B-50] Kegagalan auth/kuota di cloud harus jelas, bukan tenggelam sebagai
    # HTTPError generik. 401/402/429 punya penyebab yang sangat berbeda dan
    # tidak satu pun bisa diselesaikan dengan retry.
    if cloud and resp.status_code in (401, 402, 403, 429):
        hint = {
            401: "API key ditolak. Cek OLLAMA_CLOUD_API_KEY.",
            403: "Akses ditolak. Key mungkin belum berhak inference.",
            402: "Perlu pembayaran / plan aktif di akun Ollama Cloud.",
            429: "Rate limit / kuota Ollama Cloud terlampaui.",
        }[resp.status_code]
        raise RuntimeError(
            f"{label} HTTP {resp.status_code}: {hint} "
            f"Respons: {resp.text[:200]}"
        )

    resp.raise_for_status()
    data = resp.json()
    content = data.get("message", {}).get("content", "")
    if not content:
        raise RuntimeError(f"Respons kosong dari {label}:\n{str(data)[:300]}")

    # [B-50] Log pemakaian token juga untuk Ollama (lokal & cloud), supaya
    # perbandingan biaya/latensi antar provider apple-to-apple dengan B-49.
    pin = data.get("prompt_eval_count") or 0
    pout = data.get("eval_count") or 0
    _COST["calls"] += 1
    print(f"[Vision] {label} {_ollama_model(cloud)} | token in={pin:,} out={pout:,} "
          f"| {time.time() - t0:.0f}s | panggilan sesi: {_COST['calls']}")

    # [B-46] Ollama memotong output secara SENYAP saat kehabisan token/context.
    # Dulu ini muncul sebagai "hasil OCR kurang lengkap" tanpa petunjuk apa pun
    # di log. Sekarang dijadikan VisionTruncated supaya Phase 3 menaikkan plafon
    # dan mengulang -- sejalan dengan penanganan Gemini di B-48.
    if data.get("done_reason") == "length":
        raise VisionTruncated(
            f"Output {label} terpotong di batas token "
            f"(num_predict={max_tokens}"
            f"{'' if cloud else f', num_ctx={num_ctx}'}, terpakai={pout} token)."
        )
    return content


# ──────────────────────────────────────────────────────────────
# Gemini
# ──────────────────────────────────────────────────────────────

def _chat_gemini(messages: list, max_tokens: int, temperature: float,
                 json_mode: bool = False) -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY tidak di-set, padahal VISION_PROVIDER=gemini."
        )

    # [B-14] SDK baru `google-genai`. Paket lama `google-generativeai` sudah
    # End of Life. Import lokal supaya dependensi Gemini tetap opsional saat
    # VISION_PROVIDER=ollama.
    from google import genai
    from google.genai import types

    system_parts: list = []
    user_parts: list = []

    for msg in messages:
        role = msg.get("role", "user")
        text = (msg.get("content") or "").strip()

        # '/no_think' adalah instruksi khusus Qwen untuk menekan tag <think>.
        # Bagi Gemini itu cuma noise yang bisa mengacaukan output.
        if text.startswith("/no_think"):
            text = text[len("/no_think"):].lstrip()
        text = text.replace("/no_think\n", "").replace("/no_think", "")

        if role == "system":
            if text:
                system_parts.append(text)
            continue

        if text:
            user_parts.append(types.Part.from_text(text=text))

        for img_b64 in msg.get("images") or []:
            try:
                raw = base64.b64decode(img_b64)
            except (binascii.Error, ValueError) as e:
                raise RuntimeError(f"Gambar base64 tidak valid: {e}") from e
            user_parts.append(
                types.Part.from_bytes(data=raw, mime_type=_sniff_mime(raw))
            )

    if not user_parts:
        raise RuntimeError("Tidak ada konten user yang bisa dikirim ke Gemini.")

    client = genai.Client(api_key=api_key)

    # [B-48] MATIKAN "thinking" pada Gemini 2.5.
    #
    # Pipeline ini ditulis untuk Qwen3-VL yang tidak punya thinking. Pada
    # Gemini 2.5, thinking token:
    #   1. IKUT MEMAKAN jatah max_output_tokens -> HTML Phase 3 terpotong di
    #      tengah (persis yang terjadi pada 4.jpeg: peringatan "output Gemini
    #      terpotong di batas token" menyala, lalu hasil terpotong itu tetap
    #      dikirim);
    #   2. DITAGIH sebagai output token -- tarif termahal (~$2.50/1jt vs
    #      $0.30/1jt input).
    #
    # Untuk transkripsi OCR gambar->HTML, penalaran bertingkat tidak memberi
    # nilai tambah: modelnya menyalin apa yang terlihat. Jadi mematikan
    # thinking memperbaiki KUALITAS (tidak terpotong lagi) sekaligus MENEKAN
    # BIAYA pada saat yang sama.
    #
    # Set GEMINI_THINKING_BUDGET ke angka > 0 kalau suatu saat mau dinyalakan
    # lagi (-1 = otomatis / dikelola model).
    try:
        thinking_budget = int(os.getenv("GEMINI_THINKING_BUDGET", "0"))
    except ValueError:
        thinking_budget = 0

    # Dengan thinking mati, seluruh jatah output dipakai untuk HTML. Batas ini
    # hanya PLAFON -- penagihan mengikuti token yang benar-benar dipakai, jadi
    # melonggarkannya tidak menambah biaya, tapi mencegah pemotongan.
    try:
        out_cap = int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "0"))
    except ValueError:
        out_cap = 0
    if out_cap <= 0:
        out_cap = max_tokens * 2 if thinking_budget == 0 else max_tokens * 3

    config = types.GenerateContentConfig(
        temperature=temperature,
        max_output_tokens=out_cap,
        thinking_config=types.ThinkingConfig(thinking_budget=thinking_budget),
        # [B-45] Kunci output ke JSON valid saat pemanggil memang meminta JSON.
        response_mime_type="application/json" if json_mode else None,
        system_instruction="\n\n".join(system_parts) if system_parts else None,
        safety_settings=[
            types.SafetySetting(category=cat, threshold="BLOCK_NONE")
            for cat in (
                "HARM_CATEGORY_HARASSMENT",
                "HARM_CATEGORY_HATE_SPEECH",
                "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "HARM_CATEGORY_DANGEROUS_CONTENT",
            )
        ],
    )

    # Gemini kadang membalas 503 "high demand" atau 429 rate limit secara
    # transient. Pipeline OCR melakukan 4+ panggilan berurutan per gambar, jadi
    # satu kegagalan sesaat di tengah akan menggagalkan seluruh dokumen.
    # Retry dengan backoff supaya tidak sia-sia.
    #
    # [B-44] TAPI tidak semua 429 layak di-retry. Ada dua jenis yang sangat
    # berbeda, dan versi pertama kode ini menyamakan keduanya:
    #
    #   * per-MENIT (RequestsPerMinute...)  -> tunggu sebentar, retry berguna
    #   * per-HARI  (GenerateRequestsPerDayPerProjectPerModel-FreeTier)
    #                                      -> kuota HABIS SAMPAI BESOK.
    #                                         Retry mustahil berhasil; hanya
    #                                         membuang 14 detik per panggilan
    #                                         lalu gagal juga.
    #
    # Free tier gemini-2.5-flash hanya 20 request/HARI, sementara satu gambar
    # memakai ~6 panggilan -> habis setelah ~3 gambar. Jadi kasus per-hari ini
    # bukan kasus langka; itu kondisi normal di free tier.
    attempts = max(1, int(os.getenv("GEMINI_MAX_ATTEMPTS", "4")))
    for attempt in range(1, attempts + 1):
        try:
            response = client.models.generate_content(
                model=_gemini_model(), contents=user_parts, config=config
            )
            break
        except Exception as e:
            msg = str(e)
            code = getattr(e, "code", None) or getattr(e, "status_code", None)

            # Kuota harian habis -> gagal cepat dengan pesan yang actionable.
            if "PerDay" in msg or "per day" in msg.lower():
                raise RuntimeError(
                    "Kuota HARIAN Gemini habis (free tier gemini-2.5-flash = 20 "
                    "request/hari, dan 1 gambar memakai ~6 panggilan). Retry tidak "
                    "akan menolong sampai kuota reset.\n"
                    "  Pilihan:\n"
                    "   1. Set VISION_PROVIDER=ollama dan CLASSIFICATION_PROVIDER=ollama "
                    "di kai-dify/.env -> pakai qwen2.5vl:7b di GPU sendiri, tanpa kuota.\n"
                    "   2. Aktifkan billing di Google AI Studio.\n"
                    f"  Detail: {msg[:200]}"
                ) from e

            retryable = code in (429, 500, 502, 503, 504) or any(
                k in msg.lower() for k in ("unavailable", "high demand",
                                           "resource_exhausted", "deadline")
            )
            if not retryable or attempt == attempts:
                raise

            # Hormati retryDelay dari server kalau ada -- server tahu lebih baik
            # daripada backoff buta kita.
            wait = min(2 ** attempt, 30)
            m = re.search(r"'retryDelay':\s*'(\d+)s'", msg) or \
                re.search(r"retry in ([\d.]+)s", msg)
            if m:
                try:
                    wait = min(max(float(m.group(1)), 1.0), 120.0)
                except ValueError:
                    pass
            print(f"[Vision] Gemini {code or 'error'} transien "
                  f"(percobaan {attempt}/{attempts}), coba lagi dalam {wait:.0f}s ...")
            time.sleep(wait)

    # response.text melempar kalau tidak ada kandidat teks (mis. kena filter
    # atau berhenti karena MAX_TOKENS). Beri pesan yang menjelaskan sebabnya.
    try:
        content = (response.text or "").strip()
    except Exception as e:
        reason = None
        try:
            reason = response.candidates[0].finish_reason
        except Exception:
            pass
        raise RuntimeError(
            f"Gemini tidak mengembalikan teks (finish_reason={reason}). "
            f"Kalau MAX_TOKENS, naikkan max_tokens. Detail: {e!r}"
        ) from e

    if not content:
        raise RuntimeError("Respons kosong dari Gemini.")

    # [B-49] Log pemakaian token + estimasi biaya setiap panggilan.
    # Saldo billing terbatas, jadi biaya harus KELIHATAN, bukan ditebak dari
    # dashboard setelah habis. `thoughts` sengaja ditampilkan terpisah supaya
    # efek mematikan thinking (B-48) langsung terlihat angkanya.
    try:
        u = response.usage_metadata
        pin = getattr(u, "prompt_token_count", 0) or 0
        pout = getattr(u, "candidates_token_count", 0) or 0
        pthink = getattr(u, "thoughts_token_count", 0) or 0
        billed_out = pout + pthink
        usd = (pin * _price("GEMINI_PRICE_INPUT_PER_M", 0.30)
               + billed_out * _price("GEMINI_PRICE_OUTPUT_PER_M", 2.50))
        idr = usd * float(os.getenv("USD_TO_IDR", "16500"))
        _COST["calls"] += 1
        _COST["usd"] += usd
        print(f"[Vision] token in={pin:,} out={pout:,} thinking={pthink:,} "
              f"| ~${usd:.5f} (~Rp{idr:,.0f}) "
              f"| total sesi: {_COST['calls']} panggilan ~Rp"
              f"{_COST['usd'] * float(os.getenv('USD_TO_IDR', '16500')):,.0f}")
    except Exception:
        pass

    # [B-48] Truncation menghasilkan output yang kelihatan valid tapi terpenggal.
    # Dulu hanya diberi peringatan lalu tetap dikirim -- itu yang membuat hasil
    # 4.jpeg tidak lengkap padahal log-nya sudah memperingatkan. Sekarang
    # dianggap KEGAGALAN supaya pemanggil (Phase 3) mengulang.
    try:
        finish_reason = str(response.candidates[0].finish_reason)
    except Exception:
        finish_reason = ""
    if "MAX_TOKENS" in finish_reason:
        raise VisionTruncated(
            f"Output Gemini terpotong di batas token "
            f"(max_output_tokens={out_cap}, thinking_budget={thinking_budget}). "
            f"Naikkan GEMINI_MAX_OUTPUT_TOKENS."
        )

    return content


# ──────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────

def vision_chat(messages: list, max_tokens: int = 6000, temperature: float = 0.1,
                json_mode: bool = False, provider: str = None) -> str:
    """
    Kirim percakapan multimodal ke provider yang aktif dan kembalikan teksnya.

    `provider` boleh diisi untuk menimpa VISION_PROVIDER -- dipakai
    doc_processor/utils.py yang punya saklar sendiri (CLASSIFICATION_PROVIDER).

    `messages` memakai format ala Ollama (lihat docstring modul) apa pun
    provider-nya, sehingga call site di pipeline tidak perlu tahu provider mana
    yang sedang dipakai.

    json_mode=True meminta provider mengunci output ke JSON valid
    (Ollama: `format: "json"`, Gemini: `response_mime_type`).

    [B-45] TAPI defaultnya DIMATIKAN, dan itu berdasarkan pengukuran.
    Pada qwen2.5vl:7b, `format: "json"` memang menghasilkan JSON yang selalu
    valid, tapi membuat model jadi jauh lebih ringkas -- hasil uji Phase 1 pada
    3 gambar nyata:

        mode              field   JSON     field kunci yang hilang
        tanpa json_mode   35/35/35 rusak   -
        dengan json_mode  12/35/12 valid   complexity, has_data_table,
                                           has_signature_block, has_numbered_list,
                                           text_alignment, font_style

    Kehilangan 23 dari 35 field jauh lebih merugikan daripada koma yang hilang,
    karena field-field itu dipakai build_extraction_prompt() untuk menyusun
    prompt Phase 2 -- prompt-nya jadi salah dan hasil OCR-nya memburuk.
    Sementara koma yang hilang bisa ditambal sempurna oleh json_repair.py
    tanpa kehilangan satu field pun.

    Jadi: biarkan model bebas, lalu tambal JSON-nya. Set VISION_JSON_MODE=1
    kalau suatu saat mau mencoba jalur structured-output lagi (mis. setelah
    ganti model).
    """
    if json_mode and os.getenv("VISION_JSON_MODE", "0").strip().lower() not in ("1", "true", "yes"):
        json_mode = False

    # [B-50] `provider` bisa di-override pemanggil. Dipakai doc_processor/utils.py
    # supaya CLASSIFICATION_PROVIDER dan VISION_PROVIDER tetap dua saklar yang
    # independen, tapi berbagi SATU implementasi provider di modul ini.
    p = (provider or _provider()).strip().lower()
    if p in ("ollama", "ollama-local"):
        return _chat_ollama(messages, max_tokens, temperature, json_mode, cloud=False)
    if p in ("ollama-cloud", "ollama_cloud"):
        return _chat_ollama(messages, max_tokens, temperature, json_mode, cloud=True)
    if p == "gemini":
        return _chat_gemini(messages, max_tokens, temperature, json_mode)
    raise RuntimeError(
        f"Provider '{p}' tidak dikenal. Pilih 'gemini', 'ollama', atau 'ollama-cloud'."
    )
