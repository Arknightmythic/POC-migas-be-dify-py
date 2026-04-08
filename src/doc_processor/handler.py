# src/doc_processor/handler.py

import os
import shutil, requests
import fitz  # PyMuPDF
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

        if extension.lower() == '.pdf':
            try:
                print(f"PDF detected. Forwarding to Extractor & Chunker API as '{api_filename}'...")
                
                # Sesuaikan URL dan Port dengan tempat API baru kamu berjalan
                api_url = "http://127.0.0.1:8000/extract"

                with open(input_path, "rb") as f:
                    response = requests.post(
                        api_url,
                        # Parameter "mode" dihapus karena sudah dihandle oleh .env di server API
                        files={"file": (api_filename, f, "application/pdf")}
                    )

                if response.status_code != 200:
                    print(f"Extractor API returned error: {response.text}")
                    return
                
                # 1. Simpan PDF (Copy file asli ke output)
                shutil.copy(input_path, output_pdf_path)

                # 2. Ambil text dari JSON response dan simpan sebagai .txt
                data = response.json()
                
                # MENGUBAH KEY RESPONSE:
                # Menyesuaikan dengan key dari FastAPI kita yang baru
                cleaned_text = data.get("hasil_ekstraksi", "")
                
                if cleaned_text:
                    with open(output_txt_path, "w", encoding="utf-8") as f:
                        f.write(cleaned_text)
                    print(f"Text extracted, chunked, and saved to: {output_txt_path}")
                else:
                    print("Warning: API returned empty 'hasil_ekstraksi'")

            except Exception as e:
                print(f"Error processing PDF via external API: {e}")

        else:
            print(f"File format {extension} is not supported.")

        # Clean up the original input file
        if os.path.exists(input_path):
            os.remove(input_path)
        print(f"--- Finished processing: {current_filename} ---")