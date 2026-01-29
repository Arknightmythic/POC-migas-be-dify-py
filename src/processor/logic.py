# src/processor/logic.py

import os
import uuid
import shutil # <-- TAMBAHKAN BARIS INI
from dotenv import load_dotenv

# --- Impor kelas-kelas yang sudah Anda buat ---
from ..doc_processor.handler import DocumentProcessorHandler as OcrHandler
from ..dify_processor.runner import process_document_with_llm
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
        Menjalankan kedua proses secara berurutan dengan mengintegrasikan kelas Anda.
        """
        
        # --- Tahap 1: Proses OCR/Konversi Dokumen ---
        print(f"🔬 Memulai proses OCR untuk: {original_filename}")
        self.ocr_handler.process_document(input_path)

        processed_basename = os.path.basename(input_path)
        name, _ = os.path.splitext(processed_basename)
        processed_pdf_name = f"{name}_processed.pdf"
        ocr_output_path = os.path.join(self.ocr_handler.output_dir, processed_pdf_name)

        if not os.path.exists(ocr_output_path):
            raise FileNotFoundError(f"File PDF hasil OCR tidak ditemukan di: {ocr_output_path}")

        original_name_base, _ = os.path.splitext(original_filename)
        final_pdf_filename = f"{original_name_base}.pdf"
        final_pdf_path_on_disk = os.path.join(self.output_pdf_dir, final_pdf_filename)
        # Gunakan shutil.move untuk memindahkan file PDF juga agar konsisten
        shutil.move(ocr_output_path, final_pdf_path_on_disk)
        
        db_pdf_path = f"/legal/{final_pdf_filename}"
        print(f"✅ Proses OCR selesai. File disimpan di: {db_pdf_path}")


        # --- Tahap 2: Proses Analisis & Chunking Dify ---
        print(f"🧠 Memulai proses Dify untuk: {original_filename}")
        dify_temp_output_folder = "dify_temp_output"
        os.makedirs(dify_temp_output_folder, exist_ok=True)

        process_document_with_llm(
            input_folder=self.output_pdf_dir,
            output_folder=dify_temp_output_folder,
            file_name=final_pdf_filename,
            llm=self.dify_llm,
            dataset=DummyDifyDataset() 
        )
        
        incorrectly_named_txt_path = os.path.join(dify_temp_output_folder, f"{final_pdf_filename}.txt")
        
        if not os.path.exists(incorrectly_named_txt_path):
            print("⚠️ Proses Dify tidak menghasilkan file .txt, mungkin karena tidak ada teks.")
            return db_pdf_path, None

        final_txt_filename = f"{original_name_base}.txt"
        correctly_named_txt_path = os.path.join(dify_temp_output_folder, final_txt_filename)
        os.rename(incorrectly_named_txt_path, correctly_named_txt_path)

        try:
            print(f"Mengunggah file '{final_txt_filename}' ke Dify secara manual...")
            self.dify_dataset.upload_document_to_dataset(file_path=correctly_named_txt_path)
            print(f"Unggahan dokumen {final_txt_filename} ke dataset berhasil!")
        except Exception as e:
            print(f"Unggahan dokumen {final_txt_filename} gagal: {e}")
            return db_pdf_path, None

        # --- PERBAIKAN DI SINI ---
        # Ganti os.rename dengan shutil.move untuk memindahkan file akhir
        final_txt_path_on_disk = os.path.join(self.output_txt_dir, final_txt_filename)
        shutil.move(correctly_named_txt_path, final_txt_path_on_disk)
        # --- AKHIR PERBAIKAN ---
        
        db_txt_path = f"/legal_processed/{final_txt_filename}"
        print(f"✅ Proses Dify selesai. File disimpan di: {db_txt_path}")

        return db_pdf_path, db_txt_path 