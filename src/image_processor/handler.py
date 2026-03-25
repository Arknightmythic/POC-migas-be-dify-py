# src/image_processor/handler.py
import os
import httpx
from uuid import UUID
from pathlib import Path
from fastapi import UploadFile, HTTPException

# Menggunakan kembali logic konversi dari doc_processor
from ..doc_processor import utils as OcrUtils

# IMPORT PIPELINE BARU (tanpa memodifikasi kode aslinya)
from .doc_ocr_pipeline import run_pipeline

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
        docx_db_path = None # Variabel untuk menyimpan path DOCX
        
        try:
            # 1. Upload Raw Image ke Main BE
            raw_image_db_path = await self.upload_file_to_main_be(temp_input_path, "raw_images")
            print(f"✅ Raw image uploaded: {raw_image_db_path}")

            # 2. Ekstrak teks dengan Gemini
            text_content = OcrUtils.extract_text_with_gemini_vision(temp_input_path)
            if not text_content: raise ValueError("Text extraction failed.")

            # 3. Klasifikasi (General, Administrative, dll)
            category = OcrUtils.classify_image_content(original_filename, text_content)

            # 4. Buat PDF (Tetap jalan sebagai file default)
            base_name, _ = os.path.splitext(original_filename)
            output_filename = f"{base_name}.pdf"
            temp_pdf_path = os.path.join(self.temp_dir, output_filename)
            
            OcrUtils.create_searchable_pdf(text_content, temp_pdf_path)
            pdf_db_path = await self.upload_file_to_main_be(temp_pdf_path, "legal") 

            # -------------------------------------------------------------
            # 5. INTEGRASI: JIKA ADMINISTRATIVE, JALANKAN DOCX PIPELINE
            # -------------------------------------------------------------
            if category == "administrative":
                print(f"📄 Format Administrative terdeteksi! Menjalankan Ollama DOCX Pipeline untuk {original_filename}...")
                
                html_output = os.path.join(self.temp_dir, f"{base_name}.html")
                try:
                    # Jalankan fungsi utama dari file doc_ocr_pipeline.py
                    pipeline_result = run_pipeline(
                        image_path=temp_input_path,
                        output_path=html_output,
                        save_prompt=False,
                        wrap_page=True,
                        make_docx=True
                    )
                    
                    generated_docx = pipeline_result.get("docx_path")
                    if generated_docx and os.path.exists(generated_docx):
                        # Upload file DOCX ke Main BE
                        docx_db_path = await self.upload_file_to_main_be(generated_docx, "administrative_docs")
                        print(f"✅ DOCX berhasil diunggah: {docx_db_path}")
                        
                        # Cleanup file output pipeline (HTML dan DOCX)
                        os.remove(generated_docx)
                        if os.path.exists(html_output): os.remove(html_output)
                        
                        # Karena script doc_ocr_pipeline membuat file '_analysis.json' dan '_logo.png' 
                        # di current working directory, kita bersihkan agar folder tidak kotor
                        stem = Path(temp_input_path).stem
                        cwd = os.getcwd()
                        for suffix in ["_analysis.json", "_logo.png"]:
                            temp_artifact = os.path.join(cwd, f"{stem}{suffix}")
                            if os.path.exists(temp_artifact):
                                os.remove(temp_artifact)

                except Exception as e:
                    print(f"❌ DOCX Pipeline gagal (fallback ke PDF saja): {e}")
            # -------------------------------------------------------------

            # 6. Kirim Callback (Kirim data pdf_db_path DAN docx_db_path)
            payload = {
                "document_id": str(document_id),
                "status": "completed",
                "file_path": pdf_db_path,
                "raw_image_path": raw_image_db_path,
                "category": category,
                "docx_path": docx_db_path # Akan berisi URL jika sukses, None jika gagal/bukan administrative
            }
            await self.notify_main_api(payload)
            
            # Cleanup local temp PDF
            if os.path.exists(temp_pdf_path): os.remove(temp_pdf_path)

        except Exception as e:
            print(f"❌ Conversion failed for {original_filename}: {e}")
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
            if os.path.exists(temp_input_path):
                os.remove(temp_input_path)