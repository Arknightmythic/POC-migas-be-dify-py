# src/image_processor/handler.py
import os
import httpx
from uuid import UUID
from pathlib import Path
from fastapi import UploadFile, HTTPException

# Menggunakan kembali logic konversi dari doc_processor
from ..doc_processor import utils as OcrUtils

# IMPORT PIPELINE BARU (tanpa memodifikasi kode aslinya)
from .doc_ocr_pipeline import run_pipeline as run_digital_pipeline

try:
    from .handwritten_pipeline import run_pipeline as run_handwritten_pipeline
except ImportError:
    print("⚠️ Module handwritten_pipeline.py belum tersedia.")
    run_handwritten_pipeline = None

class ImageProcessorHandler:
    def __init__(self):
        self.temp_dir = "image_processor_temp"
        os.makedirs(self.temp_dir, exist_ok=True)
        
        self.MAIN_API_BASE_URL = os.getenv("MAIN_API_BASE_URL")
        self.callback_url = os.getenv("IMAGE_EXTRACTION_CALLBACK_URL")
        print("ImageProcessorHandler Initialized")

    async def notify_main_api(self, payload: dict):
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
        # Saran: Gunakan timeout yang sedikit lebih panjang untuk mencegah error saat sistem under-load
        async with httpx.AsyncClient() as client:
            try:
                with open(file_path, "rb") as f:
                    files = {"file": (filename, f)}
                    data = {"folder_category": folder_category}
                    print(f"🚀 Uploading {filename} to Main BE...")
                    response = await client.post(url, files=files, data=data, timeout=120.0) # Naikkan ke 120s
                    response.raise_for_status()
                    return response.json().get('saved_path')
            except Exception as e:
                # Gunakan repr(e) agar error kosong seperti ReadTimeout('') terlihat wujudnya
                print(f"❌ Upload failed for {filename}: {repr(e)}") 
                return None      

    async def convert_image_to_pdf(self, temp_input_path: str, original_filename: str, document_id: UUID):
        raw_image_db_path = None
        docx_db_path = None 
        temp_pdf_path = None
        html_output = None
        generated_docx = None
        
        # PERBAIKAN: base_name kini unik per dokumen
        base_name = f"{document_id}_{os.path.splitext(original_filename)[0]}"
        stem = Path(temp_input_path).stem
        
        try:
            # 1. Upload Raw Image ke Main BE
            raw_image_db_path = await self.upload_file_to_main_be(temp_input_path, "raw_images")
            print(f"✅ Raw image uploaded: {raw_image_db_path}")

            # 2. Ekstrak teks dengan Gemini/Ollama
            text_content = OcrUtils.extract_text_with_gemini_vision(temp_input_path)
            if not text_content: raise ValueError("Text extraction failed.")

            # 3. Klasifikasi
            category = OcrUtils.classify_image_content(original_filename, text_content)

            # 4. Buat PDF (Tidak ada redudansi baris lagi)
            output_filename = f"{base_name}.pdf"
            temp_pdf_path = os.path.join(self.temp_dir, output_filename)
            OcrUtils.create_searchable_pdf(text_content, temp_pdf_path)
            pdf_db_path = await self.upload_file_to_main_be(temp_pdf_path, "legal") 

            # 5. INTEGRASI PIPELINE
            if category == "administrative":
                print(f"📄 Format Administrative terdeteksi untuk {original_filename}!")
                hw_status = OcrUtils.classify_handwritten_status(temp_input_path)
                
                if hw_status == "handwritten" and run_handwritten_pipeline:
                    print(f"✍️ Tipe Handwritten terdeteksi! Menjalankan Handwritten Pipeline...")
                    selected_pipeline = run_handwritten_pipeline
                else:
                    print(f"🖨️ Tipe Digital terdeteksi! Menjalankan Digital Pipeline...")
                    selected_pipeline = run_digital_pipeline
                
                html_output = os.path.join(self.temp_dir, f"{base_name}.html")
                pipeline_result = selected_pipeline(
                    image_path=temp_input_path,
                    output_path=html_output,
                    save_prompt=False,
                    wrap_page=True,
                    make_docx=True
                )
                generated_docx = pipeline_result.get("docx_path")
                if generated_docx and os.path.exists(generated_docx):
                    docx_db_path = await self.upload_file_to_main_be(generated_docx, "administrative_docs")

            # 6. Kirim Callback
            payload = {
                "document_id": str(document_id),
                "status": "completed",
                "file_path": pdf_db_path,
                "raw_image_path": raw_image_db_path,
                "category": category,
                "docx_path": docx_db_path 
            }
            await self.notify_main_api(payload)

        except Exception as e:
            print(f"❌ Conversion failed for {original_filename}: {repr(e)}")
            payload = {
                "document_id": str(document_id), 
                "status": "failed", 
                "file_path": None,
                "raw_image_path": raw_image_db_path,
                "category": "general",
                "docx_path": None
            }
            await self.notify_main_api(payload)
            
        finally:
            # 🧹 PERBAIKAN CLEANUP: Hapus file temporary dengan alamat yang benar
            files_to_remove = [
                temp_input_path, 
                temp_pdf_path, 
                html_output, 
                generated_docx,
                os.path.join(self.temp_dir, f"{stem}_analysis.json"), # Memperbaiki os.getcwd()
                os.path.join(self.temp_dir, f"{stem}_logo.png")       # Memperbaiki os.getcwd()
            ]
            
            for file_path in files_to_remove:
                if file_path and os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                    except Exception as cleanup_err:
                        print(f"⚠️ Failed to clean up {file_path}: {cleanup_err}")