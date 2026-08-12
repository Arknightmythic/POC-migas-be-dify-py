# src/dify_processor/llm.py

import os

# [B-14] `google.generativeai` sudah End of Life -> pakai `google-genai`.
from google import genai
from google.genai import types
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI
from .prompt import Prompt

class LLM:
    """
    CATATAN soal embedding (verified 2026-08-10):
    `self.embedding_client` HANYA dipakai oleh SemanticChunker di
    src/dify_processor/runner.py, dan runner.py saat ini tidak dipanggil dari
    mana pun. Alur aktif meng-upload file mentah ke Dify dan Dify yang
    melakukan embedding sendiri (per-dataset, saat ini qwen3-embedding:8b via
    Ollama). Jadi mengganti embedding_model DI SINI tidak mengubah embedding
    collection Dify.
    """

    def __init__(self, provider, api_key, model_name, embedding_model):
        self.provider = (provider or "gemini").lower()
        self.model_name = model_name
        self.prompt = Prompt()

        if self.provider in ('openai', 'ollama'):
            # [B-07] Cabang 'ollama' sebelumnya TIDAK ADA, padahal .env
            # mendokumentasikannya sebagai pilihan. Akibatnya LLM_PROVIDER=ollama
            # diam-diam jatuh ke Gemini -- tanpa error, jadi sulit disadari.
            #
            # [B-08] base_url sebelumnya tidak pernah di-set, sehingga
            # OPENAI_EMBEDDING_MODEL=qwen3-embedding:8b dikirim ke
            # api.openai.com dan pasti gagal (model tidak ada di sana).
            # Ollama menyediakan endpoint OpenAI-compatible di /v1, jadi
            # keduanya bisa memakai klien yang sama, cuma beda base_url.
            if self.provider == 'ollama':
                base_url = os.getenv(
                    "OLLAMA_OPENAI_BASE_URL",
                    f"{os.getenv('OLLAMA_URL', 'http://localhost:11434').rstrip('/')}/v1",
                )
                # Ollama tidak memeriksa API key, tapi klien OpenAI menolak
                # nilai kosong -- pakai placeholder.
                api_key = api_key or "ollama"
            else:
                base_url = os.getenv("OPENAI_BASE_URL") or None
                if not api_key:
                    raise ValueError("OPENAI_API_KEY is not set.")

            self.openai_client = OpenAI(api_key=api_key, base_url=base_url)

            embedding_kwargs = {
                "model": embedding_model,
                "openai_api_key": api_key,
            }
            if base_url:
                embedding_kwargs["openai_api_base"] = base_url
                # Model non-OpenAI (mis. qwen3-embedding) tidak dikenal tiktoken;
                # minta LangChain memakai tokenizer bawaan endpoint-nya.
                embedding_kwargs["check_embedding_ctx_length"] = False
            self.embedding_client = OpenAIEmbeddings(**embedding_kwargs)

            label = "Ollama (OpenAI-compatible)" if self.provider == 'ollama' else "OpenAI"
            print(f"LLM Initialized with {label} ({self.model_name}) base_url={base_url or 'default'}")

        else:
            # Default / Fallback ke Gemini
            if not api_key:
                raise ValueError("GEMINI_API_KEY is not set.")
            # [B-14] SDK baru: Client sekali, model dipilih per-request.
            self.gemini_client = genai.Client(api_key=api_key)

            if not embedding_model.startswith("models/"):
                formatted_embedding_model = f"models/{embedding_model}"
            else:
                formatted_embedding_model = embedding_model

            self.embedding_client = GoogleGenerativeAIEmbeddings(
                model=formatted_embedding_model,
                google_api_key=api_key
            )
            print(f"LLM Initialized with Google Gemini ({self.model_name})")


    def call_llm_api(self, prompt):
        """Call LLM API based on selected provider."""
        if self.provider in ('openai', 'ollama'):
            try:
                response = self.openai_client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                print(f"Error calling {self.provider} API: {e}")
                return ""
        else:
            # Gemini Logic
            try:
                response = self.gemini_client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.0,
                        safety_settings=[
                            types.SafetySetting(category=cat, threshold="BLOCK_NONE")
                            for cat in (
                                "HARM_CATEGORY_HARASSMENT",
                                "HARM_CATEGORY_HATE_SPEECH",
                                "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                                "HARM_CATEGORY_DANGEROUS_CONTENT",
                            )
                        ],
                    ),
                )
                return (response.text or "").strip()
            except Exception as e:
                print(f"Error calling Gemini API: {e}")
                return ""

    def extract_document_info(self, chunk_text):
        """Extract document topic and chunk description using LLM."""
        document_topic = self.call_llm_api(self.prompt.chunk_topic(chunk_text))
        chunk_description = self.call_llm_api(self.prompt.chunk_description(chunk_text))
        return document_topic, chunk_description