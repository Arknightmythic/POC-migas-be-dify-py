# src/doc_processor/handler.py

import os
import shutil, requests
# [B-24] `import fitz` dihapus: tidak dipakai di file ini, dan nama `fitz`
# sudah deprecated (PyMuPDF menyarankan `import pymupdf`).
from . import utils # Import our new utils module

class DocumentProcessorHandler:
    def __init__(self):
        # Define base directories
        self.input_dir = "input_files"
        self.output_dir = "output_files"
        os.makedirs(self.input_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        print("DocumentProcessorHandler Initialized")

    def process_document(self, input_path: str, original_filename: str = None):
        """
        Main function to process a single document.
        """
        # Nama file fisik di disk (temp file dengan UUID) -> digunakan untuk generate output path yang unik
        current_filename = os.path.basename(input_path)
        name, extension = os.path.splitext(current_filename)
        
        # Output paths tetap menggunakan current_filename (UUID) agar unik dan bisa dilacak oleh logic.py
        output_pdf_path = os.path.join(self.output_dir, f"{name}_processed.pdf")
        output_txt_path = os.path.join(self.output_dir, f"{name}_processed.txt")

        # Nama file yang akan dikirim ke API (Gunakan nama asli jika ada)
        api_filename = original_filename if original_filename else current_filename

        print(f"\n--- Starting process for: {current_filename} (As: {api_filename}) ---")

        # [B-26] Lihat catatan di bawah. Di-resolve di sini supaya URL-nya
        # tersedia juga di handler error.
        base_url = os.getenv("EXTRACT_SERVICE_URL", "http://localhost:9781").rstrip("/")
        api_url = f"{base_url}/extract"

        try:
            if extension.lower() != '.pdf':
                # [B-27] Dulu ini hanya print lalu jalan terus, sehingga dokumen
                # tetap ditandai "completed". Sekarang dianggap gagal.
                raise ValueError(f"Format file {extension} tidak didukung (hanya .pdf).")

            print(f"PDF detected. Forwarding to Extractor & Chunker API as '{api_filename}'...")

            # [B-26] URL ini sebelumnya di-hardcode ke http://172.16.12.98:9781/extract.
            # Port 9781 tertutup baik di server maupun di lokal, jadi SELURUH
            # pipeline PDF mati. Service yang dimaksud adalah kai-extract
            # (POST /extract -> {"hasil_ekstraksi": ...}), yang harus
            # dijalankan sendiri:
            #   cd kai-extract && uvicorn main:app --port 9781
            with open(input_path, "rb") as f:
                # Timeout ditambahkan: sebelumnya tanpa timeout, jadi kalau
                # service extract diam saja, worker menggantung tanpa batas.
                response = requests.post(
                    api_url,
                    # Parameter "mode" dihapus karena sudah dihandle oleh .env di server API
                    files={"file": (api_filename, f, "application/pdf")},
                    timeout=600,
                )

            if response.status_code != 200:
                # [B-27] Dulu `return` biasa -> logic.py tidak menemukan output,
                # lalu dokumen tetap dilaporkan "completed" dengan file_path
                # NULL (silent data loss). Sekarang raise supaya
                # DocumentProcessorHandler.process_batch menandainya "failed".
                # [B-62] Batas dinaikkan dari 300 -> 900 karakter. Pesan
                # "ekstraksi TERPOTONG" dari kai-extract panjangnya ~300 karakter
                # dan justru bagian AKHIR-nya yang memuat cara memperbaikinya --
                # dengan batas lama, saran itu terpotong dan yang sampai ke
                # pengguna hanya keluhan tanpa jalan keluar.
                raise RuntimeError(
                    f"Extractor API ({api_url}) balas HTTP {response.status_code}: "
                    f"{response.text[:900]}"
                )

            # 1. Simpan PDF (Copy file asli ke output)
            shutil.copy(input_path, output_pdf_path)

            # 2. Ambil text dari JSON response dan simpan sebagai .txt
            data = response.json()

            # MENGUBAH KEY RESPONSE:
            # Menyesuaikan dengan key dari FastAPI kita yang baru
            cleaned_text = data.get("hasil_ekstraksi", "")

            # [B-62] kai-extract melaporkan jumlah halaman/karakter dan
            # peringatan kewajaran. Tanpa dicetak di sini, ekstraksi yang
            # mencurigakan (mis. cuma sebagian halaman tersalin) lolos tanpa
            # jejak apa pun di log pipeline.
            hal = data.get("total_halaman")
            kar = data.get("total_karakter")
            if hal is not None or kar is not None:
                print(f"Ekstraksi: {kar} karakter dari {hal} halaman, "
                      f"{data.get('total_chunks')} chunk")
            for w in (data.get("peringatan") or []):
                print(f"⚠️ kai-extract: {w}")

            if not cleaned_text:
                raise RuntimeError(
                    f"Extractor API ({api_url}) mengembalikan 'hasil_ekstraksi' kosong."
                )

            with open(output_txt_path, "w", encoding="utf-8") as f:
                f.write(cleaned_text)
            print(f"Text extracted, chunked, and saved to: {output_txt_path}")

        except requests.RequestException as e:
            # Bedakan error jaringan supaya pesannya langsung menunjuk penyebab.
            raise RuntimeError(
                f"Tidak bisa menghubungi Extractor API di {api_url}. "
                f"Pastikan service kai-extract jalan (uvicorn main:app --port 9781) "
                f"dan EXTRACT_SERVICE_URL benar. Detail: {e!r}"
            ) from e

        finally:
            # Clean up the original input file -- tetap dijalankan walau gagal,
            # supaya temp file tidak menumpuk.
            if os.path.exists(input_path):
                os.remove(input_path)
            print(f"--- Finished processing: {current_filename} ---")