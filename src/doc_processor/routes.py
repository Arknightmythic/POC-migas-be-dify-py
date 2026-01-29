# src/doc_processor/routes.py

import os
import aiofiles
from fastapi import APIRouter, BackgroundTasks, UploadFile, File, HTTPException, status
from fastapi.responses import FileResponse
from .handler import DocumentProcessorHandler

router = APIRouter()
handler = DocumentProcessorHandler()

@router.post("/process", status_code=status.HTTP_202_ACCEPTED)
async def process_document_endpoint(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """
    Accepts a document file (PDF, DOCX, JPG, PNG), saves it,
    and starts the conversion process in the background.
    """
    try:
        # Save the uploaded file asynchronously
        input_filepath = os.path.join(handler.input_dir, file.filename)
        async with aiofiles.open(input_filepath, 'wb') as out_file:
            content = await file.read()
            await out_file.write(content)

        # Add the long-running task to the background
        background_tasks.add_task(handler.process_document, input_filepath)

        processed_filename = f"{os.path.splitext(file.filename)[0]}_processed.pdf"

        return {
            "message": "File received. Processing started in the background.",
            "original_filename": file.filename,
            "output_filename": processed_filename,
            "download_endpoint": f"/api/doc-processor/download/{processed_filename}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process file: {e}")


@router.get("/download/{filename}")
async def download_processed_file(filename: str):
    """
    Downloads a processed, searchable PDF file.
    """
    output_filepath = os.path.join(handler.output_dir, filename)
    if os.path.exists(output_filepath):
        return FileResponse(path=output_filepath, media_type='application/pdf', filename=filename)
    else:
        raise HTTPException(status_code=404, detail="File not found. It may still be processing or failed to convert.")