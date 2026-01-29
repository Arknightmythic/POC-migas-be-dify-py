# src/processor/logic.py

import os
import uuid
import shutil # <-- TAMBAHKAN BARIS INI
from dotenv import load_dotenv

# --- Impor kelas-kelas yang sudah Anda buat ---
from ..doc_processor.handler import DocumentProcessorHandler as OcrHandler
# from ..dify_processor.runner import process_document_with_llm
from ..dify_processor.llm import LLM
from ..dify_processor.dify import DifyDataset

load_dotenv()

# Kelas ini memiliki metode 'upload_document_to_dataset' yang tidak melakukan apa-apa.
# Tujuannya adalah untuk "menipu" skrip runner agar tidak mengunggah file.
class DummyDifyDataset:
    def upload_document_to_dataset(self, file_path):
        print(f"Melewatkan unggahan otomatis untuk: {os.path.basename(file_path)}")
        pass

class AIServiceLogic:
    def __init__(self):
        # Inisialisasi handler OCR
        self.ocr_handler = OcrHandler()
        
        # Inisialisasi komponen untuk Dify menggunakan variabel Gemini dari .env
        self.dify_llm = LLM(
            gemini_api_key=os.getenv("GEMINI_API_KEY"),
            gemini_model=os.getenv("GEMINI_MODEL"),
            embedding_model=os.getenv("GEMINI_EMBEDDING_MODEL")
        )
        # Inisialisasi dataset Dify yang asli untuk digunakan nanti
        self.dify_dataset = DifyDataset(
            base_url=os.getenv("DATASET_BASE_URL"),
            id=os.getenv("DATASET_ID"),
            api_key=os.getenv("DATASET_API_KEY")
        )

        # Konfigurasi path dari .env
        self.MAIN_API_PUBLIC_PATH = os.getenv('MAIN_API_PUBLIC_PATH')
        self.MAIN_API_CALLBACK_URL = os.getenv('MAIN_API_CALLBACK_URL')
        if not self.MAIN_API_PUBLIC_PATH or not os.path.isdir(self.MAIN_API_PUBLIC_PATH):
            raise ValueError("MAIN_API_PUBLIC_PATH is not configured or does not exist.")
            
        self.output_pdf_dir = os.path.join(self.MAIN_API_PUBLIC_PATH, "legal")
        self.output_txt_dir = os.path.join(self.MAIN_API_PUBLIC_PATH, "legal_processed")
        os.makedirs(self.output_pdf_dir, exist_ok=True)
        os.makedirs(self.output_txt_dir, exist_ok=True)


    async def run_full_process(self, input_path: str, original_filename: str):
        """
        Menjalankan proses: Ekstraksi via API Luar -> Upload ke Dify -> Pindah ke Folder Public
        """
        
        # --- Tahap 1: Panggil Handler (API Eksternal) ---
        print(f"🔬 Memulai proses OCR (External API) untuk: {original_filename}")
        self.ocr_handler.process_document(input_path)

        # Siapkan nama file
        processed_basename = os.path.basename(input_path)
        name, _ = os.path.splitext(processed_basename)
        
        # Nama file hasil proses di folder output_files milik handler
        generated_pdf_name = f"{name}_processed.pdf"
        generated_txt_name = f"{name}_processed.txt"
        
        source_pdf_path = os.path.join(self.ocr_handler.output_dir, generated_pdf_name)
        source_txt_path = os.path.join(self.ocr_handler.output_dir, generated_txt_name)

        # --- Tahap 2: Pindahkan PDF ke Folder Public ---
        original_name_base, _ = os.path.splitext(original_filename)
        final_pdf_filename = f"{original_name_base}.pdf"
        final_pdf_path = os.path.join(self.output_pdf_dir, final_pdf_filename)

        if os.path.exists(source_pdf_path):
            shutil.move(source_pdf_path, final_pdf_path)
            db_pdf_path = f"/legal/{final_pdf_filename}"
            print(f"✅ PDF dipindahkan ke: {db_pdf_path}")
        else:
            raise FileNotFoundError(f"PDF Output tidak ditemukan: {source_pdf_path}")

        # --- Tahap 3: Handle TXT (Upload Dify & Pindah File) ---
        final_txt_filename = f"{original_name_base}.txt"
        final_txt_path = os.path.join(self.output_txt_dir, final_txt_filename)
        db_txt_path = None

        if os.path.exists(source_txt_path):
            # 3a. Upload ke Dify
            try:
                print(f"Mengunggah hasil ekstraksi '{generated_txt_name}' ke Dify...")
                self.dify_dataset.upload_document_to_dataset(file_path=source_txt_path)
                print(f"✅ Upload ke Dify berhasil!")
            except Exception as e:
                print(f"❌ Upload ke Dify gagal: {e}")
                # Kita tetap lanjut memindahkan file meskipun upload dify gagal (opsional)

            # 3b. Pindahkan TXT ke Folder Public
            shutil.move(source_txt_path, final_txt_path)
            db_txt_path = f"/legal_processed/{final_txt_filename}"
            print(f"✅ TXT dipindahkan ke: {db_txt_path}")
        else:
            print(f"⚠️ File TXT tidak ditemukan di {source_txt_path}. API mungkin gagal mengekstrak teks.")

        return db_pdf_path, db_txt_path