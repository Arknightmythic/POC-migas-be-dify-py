# src/dify_processor/llm.py

import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from .prompt import Prompt

class LLM:
    def __init__(self, gemini_api_key, gemini_model, embedding_model):
        # Konfigurasi API key untuk Gemini
        genai.configure(api_key=gemini_api_key)

        # Inisialisasi model generatif Gemini
        self.llm_client = genai.GenerativeModel(gemini_model)
        
        # --- PERBAIKAN DI SINI ---
        # Pastikan nama model embedding memiliki prefix "models/"
        # Ini adalah format yang diharapkan oleh LangChain/Google API
        if not embedding_model.startswith("models/"):
            formatted_embedding_model = f"models/{embedding_model}"
        else:
            formatted_embedding_model = embedding_model
            
        # Inisialisasi model embedding Gemini untuk LangChain
        self.embedding_client = GoogleGenerativeAIEmbeddings(
            model=formatted_embedding_model,
            google_api_key=gemini_api_key
        )
        # --- AKHIR PERBAIKAN ---

        self.prompt = Prompt()
        print("LLM Initialized with Google Gemini")

    def call_llm_api(self, prompt):
        """Call LLM using Google Gemini API."""
        try:
            # Menggunakan safety_settings untuk menghindari pemblokiran konten
            # Sesuaikan jika diperlukan
            safety_settings = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            ]
            response = self.llm_client.generate_content(
                prompt,
                generation_config={"temperature": 0.0},
                safety_settings=safety_settings
            )
            return response.text.strip()
        except Exception as e:
            print(f"Error calling Gemini API: {e}")
            # Cek jika ada respons yang diblokir
            if 'response' in locals() and hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
                print(f"Prompt blocked due to: {response.prompt_feedback.block_reason}")
            return ""

    def extract_document_info(self, chunk_text):
        """Extract document topic and chunk description using Gemini."""
        
        document_topic = self.call_llm_api(self.prompt.chunk_topic(chunk_text))
        chunk_description = self.call_llm_api(self.prompt.chunk_description(chunk_text))
        
        return document_topic, chunk_description