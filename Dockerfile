# =============================================================================
# kai-dify -- AI Document Processing Service (OCR, klasifikasi, HTML->DOCX)
#
# Perubahan penting dari versi sebelumnya:
#   - python:3.13-slim -> 3.12-slim (seluruh pengujian POC di 3.12).
#   - EXPOSE diperbaiki: dulu tertulis 9798 padahal CMD-nya 9797 -- tidak
#     konsisten dan menyesatkan.
#   - `COPY .env` DIHAPUS (secret tidak dibakar ke image; pakai env_file).
#   - Ditambah pustaka sistem untuk OpenCV. `opencv-python` butuh libGL &
#     libglib; tanpa ini `import cv2` gagal dengan
#     "ImportError: libGL.so.1: cannot open shared object file" dan SELURUH
#     pipeline gambar mati sejak import.
#   - Ditambah fonts-dejavu-core. Log lokal selalu memunculkan
#     "Warning: DejaVuSans.ttf not found. Using standard font (Helvetica)."
#     Dengan font ini terpasang, reportlab berpeluang menemukannya sehingga
#     PDF hasil OCR memakai font yang benar (aman kalau pun tidak terpakai).
#   - tzdata + TZ=Asia/Jakarta supaya log sejalan dengan jam server.
# =============================================================================
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TZ=Asia/Jakarta \
    PORT=9686

WORKDIR /app

# [B-61] Acquire::Retries -- jaringan server kantor lambat dan DNS-nya sempat
# menjawab EAI_AGAIN, membuat apt-get menggantung lalu gagal. Tanpa retry, satu
# kedipan resolver menggagalkan seluruh build image ini.
RUN apt-get -o Acquire::Retries=3 update \
    && apt-get -o Acquire::Retries=3 install -y --no-install-recommends \
        tzdata curl \
        libgl1 libglib2.0-0 \
        fonts-dejavu-core \
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY . /app/

RUN mkdir -p image_processor_temp ai_temp_input input_files output_files

EXPOSE 9686

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
