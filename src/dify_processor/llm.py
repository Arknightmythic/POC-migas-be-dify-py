# src/dify_processor/llm.py

import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI
from .prompt import Prompt

class LLM:
    def __init__(self, provider, api_key, model_name, embedding_model):
        self.provider = provider
        self.model_name = model_name
        self.prompt = Prompt()

        if self.provider == 'openai':
            # Inisialisasi OpenAI
            if not api_key:
                raise ValueError("OPENAI_API_KEY is not set.")
            self.openai_client = OpenAI(api_key=api_key)
            self.embedding_client = OpenAIEmbeddings(
                model=embedding_model,
                openai_api_key=api_key
            )
            print(f"LLM Initialized with OpenAI ({self.model_name})")
            
        else:
            # Default / Fallback ke Gemini
            if not api_key:
                raise ValueError("GEMINI_API_KEY is not set.")
            genai.configure(api_key=api_key)
            self.llm_client = genai.GenerativeModel(model_name)
            
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
        if self.provider == 'openai':
            try:
                response = self.openai_client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                print(f"Error calling OpenAI API: {e}")
                return ""
        else:
            # Gemini Logic
            try:
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
                return ""

    def extract_document_info(self, chunk_text):
        """Extract document topic and chunk description using LLM."""
        document_topic = self.call_llm_api(self.prompt.chunk_topic(chunk_text))
        chunk_description = self.call_llm_api(self.prompt.chunk_description(chunk_text))
        return document_topic, chunk_description