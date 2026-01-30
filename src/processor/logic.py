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

        separator = os.getenv("DIFY_SEPARATOR", "=== Halaman")
        max_tokens = os.getenv("DIFY_MAX_TOKENS", 2000)
        chunk_overlap = os.getenv("DIFY_CHUNK_OVERLAP", 300)

        # Inisialisasi dataset Dify yang asli untuk digunakan nanti
        self.dify_dataset = DifyDataset(
            base_url=os.getenv("DATASET_BASE_URL"),
            id=os.getenv("DATASET_ID"),
            api_key=os.getenv("DATASET_API_KEY"),
            separator=separator,
            max_tokens=max_tokens,
            chunk_overlap=chunk_overlap
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


    async def run_full_process(self, input_path: str, original_filename: str, doc_id: str):
        """
        Menjalankan proses: Ekstraksi via API Luar -> Rename -> Upload ke Dify -> Pindah ke Folder Public
        """
        
        # --- Tahap 1: Proses OCR/Konversi Dokumen ---
        print(f"🔬 Memulai proses OCR untuk: {original_filename}")
        self.ocr_handler.process_document(input_path)

        processed_basename = os.path.basename(input_path)
        name, _ = os.path.splitext(processed_basename)
        
        # Nama file output raw dari handler OCR
        generated_pdf_name = f"{name}_processed.pdf"
        generated_txt_name = f"{name}_processed.txt"
        
        source_pdf_path = os.path.join(self.ocr_handler.output_dir, generated_pdf_name)
        source_txt_path = os.path.join(self.ocr_handler.output_dir, generated_txt_name)

        # Siapkan nama file akhir yang diinginkan: {NamaFile}-{UUID}
        original_name_base, _ = os.path.splitext(original_filename)
        
        final_pdf_filename = f"{original_name_base}.pdf" # PDF tetap nama file asli
        final_txt_filename = f"{original_name_base}-{doc_id}.txt" # TXT pakai UUID

        # --- Tahap 2: Pindahkan PDF ke Folder Public ---
        final_pdf_path = os.path.join(self.output_pdf_dir, final_pdf_filename)

        if os.path.exists(source_pdf_path):
            shutil.move(source_pdf_path, final_pdf_path)
            db_pdf_path = f"/legal/{final_pdf_filename}"
            print(f"✅ PDF dipindahkan ke: {db_pdf_path}")
        else:
            raise FileNotFoundError(f"PDF Output tidak ditemukan: {source_pdf_path}")

        # --- Tahap 3: Handle TXT (Rename -> Upload Dify -> Pindah File) ---
        final_txt_path = os.path.join(self.output_txt_dir, final_txt_filename)
        db_txt_path = None

        if os.path.exists(source_txt_path):
            # 3a. RENAME file di folder sementara TERLEBIH DAHULU
            # Agar saat di-upload ke Dify, namanya sudah benar.
            temp_renamed_txt_path = os.path.join(self.ocr_handler.output_dir, final_txt_filename)
            os.rename(source_txt_path, temp_renamed_txt_path)
            
            # 3b. Upload ke Dify (menggunakan file yang sudah di-rename)
            try:
                print(f"Mengunggah hasil ekstraksi '{final_txt_filename}' ke Dify...")
                self.dify_dataset.upload_document_to_dataset(file_path=temp_renamed_txt_path)
                print(f"✅ Upload ke Dify berhasil!")
            except Exception as e:
                print(f"❌ Upload ke Dify gagal: {e}")

            # 3c. Pindahkan TXT ke Folder Public
            # Kita memindahkan temp_renamed_txt_path karena source_txt_path sudah tidak ada (sudah direname)
            shutil.move(temp_renamed_txt_path, final_txt_path)
            
            db_txt_path = f"/legal_processed/{final_txt_filename}"
            print(f"✅ TXT dipindahkan ke: {db_txt_path}")
        else:
            print(f"⚠️ File TXT tidak ditemukan di {source_txt_path}. API mungkin gagal mengekstrak teks.")

        return db_pdf_path, db_txt_path