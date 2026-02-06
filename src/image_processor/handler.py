# src/image_processor/handler.py
import os
import httpx
from uuid import UUID
from fastapi import UploadFile, HTTPException

# Menggunakan kembali logic konversi dari doc_processor
from ..doc_processor import utils as OcrUtils

class ImageProcessorHandler:
    def __init__(self):
        # Direktori sementara di dalam AI service
        self.temp_dir = "image_processor_temp"
        os.makedirs(self.temp_dir, exist_ok=True)
        
        self.MAIN_API_BASE_URL = os.getenv("MAIN_API_BASE_URL")
        # URL Callback ke backend utama dari .env
        self.callback_url = os.getenv("IMAGE_EXTRACTION_CALLBACK_URL")
        print("ImageProcessorHandler Initialized")

    async def notify_main_api(self, payload: dict):
        """Kirim status kembali ke API Utama."""
        if not self.callback_url:
            print("⚠️ IMAGE_EXTRACTION_CALLBACK_URL not set. Cannot send status update.")
            return

        async with httpx.AsyncClient() as client:
            try:
                print(f"🚀 Sending callback to {self.callback_url} with payload: {payload}")
                response = await client.post(self.callback_url, json=payload, timeout=60)
                response.raise_for_status()
                print(f"✅ Callback sent successfully for doc {payload.get('document_id')}")
            except httpx.RequestError as e:
                print(f"❌ Failed to send callback for doc {payload.get('document_id')}: {e}")

    async def upload_file_to_main_be(self, file_path: str, folder_category: str):
        if not os.path.exists(file_path): return None
        url = f"{self.MAIN_API_BASE_URL}/api/sistem-documents/store-file"
        filename = os.path.basename(file_path)
        async with httpx.AsyncClient() as client:
            try:
                with open(file_path, "rb") as f:
                    files = {"file": (filename, f)}
                    data = {"folder_category": folder_category}
                    print(f"🚀 Uploading {filename} to Main BE...")
                    response = await client.post(url, files=files, data=data, timeout=60)
                    response.raise_for_status()
                    return response.json().get('saved_path')
            except Exception as e:
                print(f"❌ Upload failed: {e}")
                return None        

    async def convert_image_to_pdf(self, temp_input_path: str, original_filename: str, document_id: UUID):
        raw_image_db_path = None
        try:
           # 1. Upload Raw Image ke Main BE (Ganti shutil.copy)
            # Kita perlu kategori 'raw_images' di Main BE handler
            raw_image_db_path = await self.upload_file_to_main_be(temp_input_path, "raw_images")
            print(f"✅ Raw image uploaded: {raw_image_db_path}")

            # 2. Ekstrak teks (Logic OcrUtils tetap sama karena baca file temp lokal)
            text_content = OcrUtils.extract_text_with_gemini_vision(temp_input_path)
            if not text_content: raise ValueError("Text extraction failed.")

            # 3. Klasifikasi
            category = OcrUtils.classify_image_content(original_filename, text_content)

            # 4. Buat PDF di folder TEMP lokal AI Service
            base_name, _ = os.path.splitext(original_filename)
            output_filename = f"{base_name}.pdf"
            temp_pdf_path = os.path.join(self.temp_dir, output_filename) # Simpan di temp lokal dulu
            
            OcrUtils.create_searchable_pdf(text_content, temp_pdf_path)
            
            # 5. Upload PDF ke Main BE
            # Tentukan kategori folder berdasarkan hasil klasifikasi
            # Jika 'general', mungkin masuk ke 'legal' atau folder khusus 'extract_results'
            # Di Main BE handler folder_map perlu disesuaikan jika ingin dinamis. 
            # Untuk simplifikasi, kita masukkan ke 'legal' saja atau buat logic mapping di Main BE.
            
            # Kita pakai 'legal' untuk semua hasil PDF image extraction agar konsisten
            pdf_db_path = await self.upload_file_to_main_be(temp_pdf_path, "legal") 

            # 6. Kirim Callback
            payload = {
                "document_id": str(document_id),
                "status": "completed",
                "file_path": pdf_db_path,
                "raw_image_path": raw_image_db_path,
                "category": category
            }
            await self.notify_main_api(payload)
            
            # Cleanup local temp PDF
            if os.path.exists(temp_pdf_path): os.remove(temp_pdf_path)
            

        except Exception as e:
            print(f"❌ Conversion failed for {original_filename}: {e}")
            # Kirim callback gagal
            payload = {
                "document_id": str(document_id), 
                "status": "failed", 
                "file_path": None,
                "raw_image_path": raw_image_db_path, # Kirim path gambar mentah jika sudah tersimpan
                "category": "general"
            }
            await self.notify_main_api(payload)
        finally:
            if os.path.exists(temp_input_path):
                os.remove(temp_input_path)