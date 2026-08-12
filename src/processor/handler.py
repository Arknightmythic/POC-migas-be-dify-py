# src/processor/handler.py

import asyncio
import os
import shutil
import httpx
from typing import List, Dict
from uuid import UUID
from .logic import AIServiceLogic
# --- TAMBAHKAN BARIS INI ---
from fastapi import HTTPException, status

class DocumentProcessorHandler:
    def __init__(self):
        self.logic = AIServiceLogic()
        self.input_dir = "ai_temp_input"
        os.makedirs(self.input_dir, exist_ok=True)
        print("DocumentProcessorHandler Initialized")

    async def delete_document(self, document_name: str):
        """Menghapus dokumen dari Dify dataset."""
        print(f"Received request to delete document from Dify: {document_name}")
        try:
            success = self.logic.dify_dataset.delete_document_from_dataset(document_name)
            if success:
                return {"status": "success", "message": f"Document '{document_name}' deleted from Dify."}
            else:
                return {"status": "not_found", "message": f"Document '{document_name}' not found in Dify or failed to delete."}
        except Exception as e:
            print(f"Error during Dify deletion process: {e}")
            raise HTTPException(status_code=500, detail="An internal error occurred during Dify deletion.")

    async def notify_main_api(self, doc_id: UUID, status: str, pdf_path: str = None, txt_path: str = None):
        """
        Kirim status kembali ke API Utama, dengan RETRY.

        [B-43] Masalahnya sama seperti di image_processor/handler.py: callback
        fire-and-forget, sekali gagal dokumen menggantung selamanya walaupun
        pemrosesan sukses. Di sini bahkan lebih parah -- response-nya tidak
        pernah diperiksa sama sekali (`await client.post(...)` tanpa
        raise_for_status), jadi kai-be bisa menjawab 500 dan sisi ini tetap
        menganggap berhasil.
        """
        if not self.logic.MAIN_API_CALLBACK_URL:
            print("❌ MAIN_API_CALLBACK_URL not set. Cannot send status update.")
            return

        payload = {
            "document_id": str(doc_id),
            "status": status,
            "searchable_pdf_path": pdf_path,
            "processed_text_path": txt_path,
        }

        attempts = max(1, int(os.getenv("CALLBACK_MAX_ATTEMPTS", "5")))
        timeout = float(os.getenv("CALLBACK_TIMEOUT", "120"))

        for attempt in range(1, attempts + 1):
            try:
                print(f"🚀 Sending callback ({attempt}/{attempts}) for doc {doc_id} "
                      f"with status: {status}")
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        self.logic.MAIN_API_CALLBACK_URL, json=payload, timeout=timeout
                    )
                    response.raise_for_status()
                print(f"✅ Callback sent successfully for doc {doc_id}")
                return
            except Exception as e:
                if attempt == attempts:
                    print(
                        f"❌❌ CALLBACK GAGAL PERMANEN untuk doc {doc_id} setelah "
                        f"{attempts} percobaan: {repr(e)}\n"
                        f"     Status dokumen ini akan MENGGANTUNG di database.\n"
                        f"     Payload yang gagal terkirim (untuk kirim ulang manual):\n"
                        f"     {payload}"
                    )
                    return
                wait = min(2 ** attempt, 30)
                print(f"⚠️ Callback doc {doc_id} gagal ({repr(e)}), "
                      f"coba lagi dalam {wait}s ...")
                await asyncio.sleep(wait)

    async def process_batch(self, tasks: List[Dict]):
        """Process files from a list of tasks containing saved file paths."""
        for task in tasks:
            temp_input_path = task.get("temp_path")
            original_filename = task.get("original_filename")
            doc_id = task.get("doc_id")

            if not all([temp_input_path, original_filename, doc_id]):
                print(f"⚠️ Warning: Invalid task data received: {task}. Skipping.")
                continue

            try:
                print(f"\n--- Starting AI processing for: {original_filename} (ID: {doc_id}) ---")
                
                # --- PERUBAHAN DI SINI: Passing doc_id ke run_full_process ---
                pdf_path, txt_path = await self.logic.run_full_process(
                    input_path=temp_input_path, 
                    original_filename=original_filename,
                    doc_id=str(doc_id)
                )
                
                await self.notify_main_api(doc_id, "completed", pdf_path, txt_path)

            except Exception as e:
                print(f"❌ Full process failed for {original_filename}: {e}")
                await self.notify_main_api(doc_id, "failed")