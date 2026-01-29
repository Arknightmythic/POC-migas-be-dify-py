# main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from dotenv import load_dotenv
# --- PERBAIKAN DI SINI ---
# Impor router yang benar dari src/processor/routes.py
from src.processor.routes import router as processor_router
from src.image_processor.routes import router as image_processor_router
# Load environment variables from .env file
load_dotenv()

class AIServiceAPI: # Ganti nama kelas agar sesuai
    def __init__(self):
        self.app = FastAPI(
            title="AI Document Processing Service",
            description="An API to process documents with OCR and Dify logic.",
            version="1.0.0"
        )
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        self.include_routers()

    def include_routers(self):
        # --- PERBAIKAN DI SINI ---
        # Gunakan router yang benar dan hapus prefix agar URL-nya menjadi http://localhost:9798/process-batch
        self.app.include_router(processor_router, prefix="", tags=["Document Processing"])
        self.app.include_router(image_processor_router, prefix="/image-processor", tags=["Image to TXT Conversion"])
    def run(self):
        uvicorn.run(
            self.app,
            host="0.0.0.0", # Listen on all available network interfaces
            port=9797, # Port yang benar adalah 9798 sesuai rencana kita
        )

ai_service_api = AIServiceAPI() # Ganti nama instance
app = ai_service_api.app

if __name__ == "__main__":
    ai_service_api.run()