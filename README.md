# kai-dify — AI Document Processing Service

OCR dokumen, klasifikasi, pipeline HTML→DOCX, dan upload ke dataset Dify.
Dipanggil oleh `kai-be` lewat `AI_SERVICE_URL` (default `http://localhost:9797`).

## Installation

```bash
python -m venv .venv
```

install requirements (pakai python milik venv, bukan `pip` dari PATH):

```bash
# Windows
.venv/Scripts/python.exe -m pip install -r requirements.txt
# Linux / macOS
.venv/bin/python -m pip install -r requirements.txt
```

## Menjalankan

```bash
# Windows
.venv/Scripts/python.exe main.py
# atau
.venv/Scripts/python.exe -m uvicorn main:app --port 9797
```

> ⚠️ **Pakai bentuk `python -m uvicorn`, jangan `uvicorn` langsung.**
> Repo ini berdampingan dengan `kai-be` dan `kai-extract` yang punya venv
> sendiri. Kalau salah satu venv itu masih aktif di shell (`VIRTUAL_ENV`
> ter-set), `uvicorn` di PATH akan menunjuk venv yang salah dan gagal dengan
> `ModuleNotFoundError` untuk paket yang jelas sudah terinstall.
>
> Kalau perlu meng-activate: `deactivate` dulu, lalu
> **`source .venv/Scripts/activate`** (tanpa `source`, PATH shell induk tidak
> berubah).

Atau pakai docker:

```bash
docker compose up -d
```

Note: don't forget to change `localhost` to `host.docker.internal` for local development!

## Dependensi service lain

| Butuh | Untuk | Config |
|---|---|---|
| **kai-extract** di port 9781 | ekstraksi PDF (`POST /extract`) | `EXTRACT_SERVICE_URL` |
| **kai-be** di port 9798 | callback status + simpan file | `MAIN_API_BASE_URL`, `*_CALLBACK_URL` |
| **Dify** | dataset knowledge base | `DATASET_BASE_URL`, `DATASET_ID`, `QNA_DATASET_ID` |
| Gemini / Ollama | LLM, OCR, klasifikasi, vision | `LLM_PROVIDER`, `CLASSIFICATION_PROVIDER`, `VISION_PROVIDER` |