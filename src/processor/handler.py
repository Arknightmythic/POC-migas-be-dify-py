# src/processor/handler.py

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

    # --- TAMBAHKAN KODE DI BAWAH INI ---
    async def delete_document(self, document_name: str):
        """Menghapus dokumen dari Dify dataset."""
        print(f"Received request to delete document from Dify: {document_name}")
        try:
            # Panggil metode delete dari logic.dify_dataset secara langsung
            success = self.logic.dify_dataset.delete_document_from_dataset(document_name)
            if success:
                return {"status": "success", "message": f"Document '{document_name}' deleted from Dify."}
            else:
                return {"status": "not_found", "message": f"Document '{document_name}' not found in Dify or failed to delete."}
        except Exception as e:
            print(f"Error during Dify deletion process: {e}")
            raise HTTPException(status_code=500, detail="An internal error occurred during Dify deletion.")
    # --- BATAS AKHIR PENAMBAHAN KODE ---

    async def notify_main_api(self, doc_id: UUID, status: str, pdf_path: str = None, txt_path: str = None):
        """Kirim status kembali ke API Utama."""
        if not self.logic.MAIN_API_CALLBACK_URL:
            print("❌ MAIN_API_CALLBACK_URL not set. Cannot send status update.")
            return

        payload = {
            "document_id": str(doc_id),
            "status": status,
            "searchable_pdf_path": pdf_path,
            "processed_text_path": txt_path,
        }
        
        async with httpx.AsyncClient() as client:
            try:
                print(f"🚀 Sending callback for doc {doc_id} with status: {status}")
                await client.post(self.logic.MAIN_API_CALLBACK_URL, json=payload, timeout=60)
            except httpx.RequestError as e:
                print(f"❌ Failed to send callback for doc {doc_id}: {e}")

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
                
                pdf_path, txt_path = await self.logic.run_full_process(temp_input_path, original_filename=original_filename)
                
                await self.notify_main_api(doc_id, "completed", pdf_path, txt_path)

            except Exception as e:
                print(f"❌ Full process failed for {original_filename}: {e}")
                await self.notify_main_api(doc_id, "failed")