import os
import httpx
from dotenv import load_dotenv

# --- Impor kelas-kelas yang sudah Anda buat ---
from ..doc_processor.handler import DocumentProcessorHandler as OcrHandler
from ..dify_processor.llm import LLM
from ..dify_processor.dify import DifyDataset

load_dotenv()

class AIServiceLogic:
    def __init__(self):
        # Inisialisasi handler OCR
        self.ocr_handler = OcrHandler()
        
       # --- PERUBAHAN: Setup LLM Dinamis ---
        # [B-07] Sebelumnya pemilihan hanya dua arah (openai vs "selain itu"),
        # sehingga LLM_PROVIDER=ollama diam-diam memakai kredensial Gemini.
        # Sekarang tiga provider dipetakan eksplisit.
        self.llm_provider = os.getenv("LLM_PROVIDER", "gemini").lower()

        if self.llm_provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            model_name = os.getenv("OPENAI_MODEL")
            embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL")
        elif self.llm_provider == "ollama":
            # Ollama dilayani lewat endpoint OpenAI-compatible (lihat LLM.__init__).
            api_key = os.getenv("OLLAMA_API_KEY", "ollama")
            model_name = os.getenv("OLLAMA_MODEL") or os.getenv("OLLAMA_VISION_MODEL")
            embedding_model = os.getenv("OLLAMA_EMBEDDING_MODEL", "qwen3-embedding:8b")
        else:
            api_key = os.getenv("GEMINI_API_KEY")
            model_name = os.getenv("GEMINI_MODEL")
            embedding_model = os.getenv("GEMINI_EMBEDDING_MODEL")

        self.dify_llm = LLM(
            provider=self.llm_provider,
            api_key=api_key,
            model_name=model_name,
            embedding_model=embedding_model
        )

        separator = os.getenv("DIFY_SEPARATOR", "=== Halaman")
        max_tokens = os.getenv("DIFY_MAX_TOKENS", 2000)
        chunk_overlap = os.getenv("DIFY_CHUNK_OVERLAP", 300)

        # Inisialisasi dataset Dify
        self.dify_dataset = DifyDataset(
            base_url=os.getenv("DATASET_BASE_URL"),
            id=os.getenv("DATASET_ID"),
            api_key=os.getenv("DATASET_API_KEY"),
            separator=separator,
            max_tokens=max_tokens,
            chunk_overlap=chunk_overlap
        )

        qna_dataset_id = os.getenv("QNA_DATASET_ID")

        self.qna_dataset = DifyDataset(
            base_url=os.getenv("DATASET_BASE_URL"), # Asumsi base URL sama
            id=qna_dataset_id,
            api_key=os.getenv("DATASET_API_KEY"), # Asumsi API Key sama
            separator=separator,
            max_tokens=max_tokens,
            chunk_overlap=chunk_overlap
        )

        # Konfigurasi path dari .env
        self.MAIN_API_BASE_URL = os.getenv('MAIN_API_BASE_URL')
        self.MAIN_API_CALLBACK_URL = os.getenv('MAIN_API_CALLBACK_URL')
        
        if not self.MAIN_API_BASE_URL:
             print("⚠️ MAIN_API_BASE_URL not set. File transfer might fail.")

    # --- HELPER FUNGSI UPLOAD ---
    async def upload_file_to_main_be(self, file_path: str, folder_category: str):
        """Mengirim file lokal ke Main Backend via API."""
        if not os.path.exists(file_path):
            print(f"❌ File not found for upload: {file_path}")
            return None

        # Pastikan URL valid
        if not self.MAIN_API_BASE_URL:
            print("❌ MAIN_API_BASE_URL not configured.")
            return None

        url = f"{self.MAIN_API_BASE_URL}/api/sistem-documents/store-file"
        filename = os.path.basename(file_path)
        
        async with httpx.AsyncClient() as client:
            try:
                with open(file_path, "rb") as f:
                    files = {"file": (filename, f)}
                    data = {"folder_category": folder_category}
                    print(f"🚀 Uploading {filename} to Main BE ({folder_category})...")
                    response = await client.post(url, files=files, data=data, timeout=120) # Timeout dinaikkan
                    response.raise_for_status()
                    result = response.json()
                    print(f"✅ Upload success: {result.get('saved_path')}")
                    return result.get('saved_path')
            except Exception as e:
                print(f"❌ Failed to upload file to Main BE: {e}")
                return None


    async def run_full_process(self, input_path: str, original_filename: str, doc_id: str):
        """
        Menjalankan proses: Ekstraksi via API Luar -> Rename ke Nama Asli -> Upload ke Dify -> Simpan dengan UUID
        """
        
        # --- Tahap 1: Proses OCR/Konversi Dokumen ---
        print(f"🔬 Memulai proses OCR untuk: {original_filename}")
        
        # Proses OCR (Output file akan ada di self.ocr_handler.output_dir)
        self.ocr_handler.process_document(input_path, original_filename=original_filename)

        processed_basename = os.path.basename(input_path)
        name, _ = os.path.splitext(processed_basename)
        
        generated_pdf_name = f"{name}_processed.pdf"
        generated_txt_name = f"{name}_processed.txt"
        
        source_pdf_path = os.path.join(self.ocr_handler.output_dir, generated_pdf_name)
        source_txt_path = os.path.join(self.ocr_handler.output_dir, generated_txt_name)

        # Siapkan nama file akhir
        original_name_base, _ = os.path.splitext(original_filename)
        final_pdf_filename = f"{original_name_base}.pdf"
        final_txt_filename = f"{original_name_base}-{doc_id}.txt" 

        # --- Tahap 2: Upload PDF ke Main BE ---
        db_pdf_path = None
        if os.path.exists(source_pdf_path):
            temp_pdf_path = os.path.join(self.ocr_handler.output_dir, final_pdf_filename)
            if os.path.exists(temp_pdf_path): os.remove(temp_pdf_path)
            os.rename(source_pdf_path, temp_pdf_path)
            
            db_pdf_path = await self.upload_file_to_main_be(temp_pdf_path, "legal")
            
            if db_pdf_path and os.path.exists(temp_pdf_path):
                os.remove(temp_pdf_path)
        else:
             print(f"⚠️ PDF Output tidak ditemukan: {source_pdf_path}")

        # --- Tahap 3: Handle TXT (Dify & Upload) ---
        db_txt_path = None
        if os.path.exists(source_txt_path):
            temp_txt_path = os.path.join(self.ocr_handler.output_dir, final_txt_filename)
            if os.path.exists(temp_txt_path): os.remove(temp_txt_path)
            os.rename(source_txt_path, temp_txt_path)
            
            # Upload ke Dify (DATASET UTAMA)
            try:
                print(f"Mengunggah ke Dify (Main Dataset)...")
                self.dify_dataset.upload_document_to_dataset(file_path=temp_txt_path)
            except Exception as e:
                print(f"❌ Upload ke Dify gagal: {e}")

            # Upload ke Main BE via API
            db_txt_path = await self.upload_file_to_main_be(temp_txt_path, "legal_processed")

            if os.path.exists(temp_txt_path):
                os.remove(temp_txt_path)
        else:
             print(f"⚠️ File TXT tidak ditemukan.")

        return db_pdf_path, db_txt_path