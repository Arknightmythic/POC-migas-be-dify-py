# src/processor/routes.py

# --- TAMBAHKAN 'status' DI BARIS INI ---
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks, status
from typing import List, Dict
from .handler import DocumentProcessorHandler
from uuid import UUID
import os
import aiofiles

router = APIRouter()
handler = DocumentProcessorHandler()

# --- TAMBAHKAN KODE DI BAWAH INI ---
@router.delete("/delete-document/{document_name}", status_code=status.HTTP_200_OK)
async def delete_dify_document(document_name: str):
    """
    Menghapus sebuah dokumen dari dataset Dify berdasarkan namanya.
    Nama dokumen harus sama persis dengan yang ada di Dify (e.g., 'mydoc.txt').
    """
    return await handler.delete_document(document_name)
# --- BATAS AKHIR PENAMBAHAN KODE ---

@router.post("/process-batch")
async def process_document_batch(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    document_ids: List[str] = Form(...)
):
    if len(files) != len(document_ids):
        raise HTTPException(status_code=400, detail="The number of files and document_ids must be the same.")

    tasks_to_run = []
    for file, doc_id_str in zip(files, document_ids):
        temp_file_path = os.path.join(handler.input_dir, f"{UUID(doc_id_str)}-{file.filename}")
        
        try:
            async with aiofiles.open(temp_file_path, 'wb') as out_file:
                content = await file.read()
                await out_file.write(content)
            
            tasks_to_run.append({
                "temp_path": temp_file_path,
                "original_filename": file.filename,
                "doc_id": UUID(doc_id_str)
            })
        except Exception as e:
            print(f"Error saving file {file.filename}: {e}")
        finally:
            await file.close()

    if tasks_to_run:
        background_tasks.add_task(handler.process_batch, tasks_to_run)
    
    return {
        "message": f"Accepted {len(tasks_to_run)} files for processing. This will continue in the background."
    }

@router.post("/knowledge/upload-qna")
async def upload_qna_document(
    file: UploadFile = File(...)
):
    """
    Endpoint khusus untuk menerima file QnA (.txt) dari Main BE 
    dan langsung menguploadnya ke Dify Dataset (QnA).
    """
    if not file.filename.endswith('.txt'):
        raise HTTPException(status_code=400, detail="Only .txt files are allowed for QnA upload.")

    # Simpan sementara
    temp_path = os.path.join(handler.input_dir, file.filename)
    try:
        async with aiofiles.open(temp_path, 'wb') as out_file:
            content = await file.read()
            await out_file.write(content)
        
        # Upload langsung ke Dify (DATASET QnA)
        try:
            print(f"Uploading QnA file {file.filename} to Dify (QnA Dataset)...")
            
            # --- PERUBAHAN DI SINI: Gunakan qna_dataset ---
            handler.logic.qna_dataset.upload_document_to_dataset(file_path=temp_path)
            
            print(f"✅ QnA Upload Success: {file.filename}")
            return {"message": "QnA uploaded successfully to Dify"}
        except Exception as e:
            print(f"❌ Dify Upload Error: {e}")
            raise HTTPException(status_code=500, detail=f"Dify upload failed: {e}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {e}")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)