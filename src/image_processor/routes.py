# src/image_processor/routes.py
import os
import aiofiles
from fastapi import APIRouter, BackgroundTasks, UploadFile, File, Form
from uuid import UUID
from .handler import ImageProcessorHandler

router = APIRouter()
handler = ImageProcessorHandler()

@router.post("/process-image")
async def process_image(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    document_id: UUID = Form(...)
):
    """
    Endpoint khusus untuk menerima gambar, mengubahnya menjadi PDF,
    menyimpannya di direktori publik, dan mengirim callback.
    """
    # Buat path sementara yang unik untuk menyimpan file
    temp_input_path = os.path.join(handler.temp_dir, f"{document_id}-{file.filename}")

    # Simpan file yang diunggah ke path sementara SECARA LANGSUNG
    try:
        async with aiofiles.open(temp_input_path, 'wb') as out_file:
            content = await file.read()
            await out_file.write(content)
    finally:
        await file.close()
    
    # Jadwalkan background task dengan PATH file, bukan objek file
    background_tasks.add_task(
        handler.convert_image_to_pdf, 
        temp_input_path=temp_input_path, 
        original_filename=file.filename,
        document_id=document_id
    )

    return {"message": f"Processing started for {file.filename} in the background."}