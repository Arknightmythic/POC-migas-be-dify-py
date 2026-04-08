import os
import base64
import fitz  # PyMuPDF
import re
import requests
import google.generativeai as genai
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.enums import TA_LEFT
from reportlab.platypus import Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from docx2pdf import convert
from dotenv import load_dotenv
import mimetypes # Tambahkan ini untuk mendeteksi tipe file gambar
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
GEMINI_MODEL = os.getenv('GEMINI_MODEL')

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-4o-mini')

# --- KONFIGURASI TOGGLE BARU ---
CLASSIFICATION_PROVIDER = os.getenv('CLASSIFICATION_PROVIDER', 'gemini').lower()
OLLAMA_URL = os.getenv('OLLAMA_URL', 'http://localhost:11434')
OLLAMA_VISION_MODEL = os.getenv('OLLAMA_VISION_MODEL', 'qwen3-vl:8b-instruct-bf16')

# --- KEMBALIKAN FUNGSI INI ---
def image_to_base64(image_path: str):
    """Membaca file gambar dan mengubahnya menjadi string Base64."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"Gagal mengubah gambar ke Base64: {e}")
        return None
    
def call_openai_vision_or_text(prompt: str, base64_image: str = None, mime_type: str = 'image/png') -> str:
    if not OPENAI_API_KEY:
        print("⚠️ OPENAI_API_KEY tidak ditemukan!")
        return ""
        
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    # Format messages standard
    messages = [{"role": "user", "content": []}]
    messages[0]["content"].append({"type": "text", "text": prompt})
    
    # Jika ada gambar, tambahkan objek image_url
    if base64_image:
        messages[0]["content"].append({
            "type": "image_url",
            "image_url": {
                "url": f"data:{mime_type};base64,{base64_image}"
            }
        })

    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            temperature=0.1
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"❌ Error saat menghubungi OpenAI: {e}")
        return ""
    
def call_ollama_classification(prompt: str, base64_image: str = None) -> str:
    """Helper untuk memanggil Ollama API (Text atau Vision)."""
    # Tambahkan instruksi /no_think agar Qwen tidak mengeluarkan tag <think>
    if "/no_think" not in prompt:
        prompt = f"/no_think\n{prompt}"
        
    messages = [{"role": "user", "content": prompt}]
    
    # Jika ada gambar, sisipkan ke dalam request
    if base64_image:
        messages[0]["images"] = [base64_image]

    payload = {
        "model": OLLAMA_VISION_MODEL,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": 0.1 # Dibuat rendah agar hasilnya deterministik
        }
    }

    try:
        url = f"{OLLAMA_URL.rstrip('/')}/api/chat"
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        result_text = response.json().get("message", {}).get("content", "").strip()
        return result_text
    except Exception as e:
        print(f"❌ Error saat menghubungi Ollama: {e}")
        return ""

# --- UBAH FUNGSI INI UNTUK MENDUKUNG TOGGLE OLLAMA/GEMINI ---
def extract_text_with_gemini_vision(image_path: str):
    """Mengirim gambar (sebagai Base64) ke model AI (Gemini atau Ollama) untuk ekstraksi teks."""
    
    print(f"Mengubah {os.path.basename(image_path)} ke Base64...")
    base64_image = image_to_base64(image_path)
    if not base64_image:
        return ""
    
    # Dapatkan tipe MIME dari file gambar
    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type is None:
        print(f"Tidak dapat mendeteksi tipe MIME untuk {image_path}. Menggunakan default 'image/png'.")
        mime_type = 'image/png'

    prompt = """
            Transcribe the text from this image with high accuracy. 
            Rules:
            1. Fix broken lines: If a sentence is broken across multiple lines in the image, join them into a single continuous line.
            2. Preserve structure: Keep distinct paragraphs and numbered lists (1., 2., etc.) on separate lines.
            3. Output format: Return clean text where each bullet point or paragraph is on its own line.
            4. Do not include markdown code blocks (```).
            """

    # --- JIKA TOGGLE MENGGUNAKAN OLLAMA ---
    if CLASSIFICATION_PROVIDER == 'openai':
        print(f"Menghubungi OpenAI ({OPENAI_MODEL}) untuk OCR...")
        return call_openai_vision_or_text(prompt, base64_image, mime_type)
    
    elif CLASSIFICATION_PROVIDER == 'ollama':
        print(f"Menghubungi Ollama ({OLLAMA_VISION_MODEL}) untuk mengekstrak teks OCR dari gambar...")
        try:
            # Gunakan helper yang sama dengan yang kita buat untuk klasifikasi
            result = call_ollama_classification(prompt, base64_image)
            print("Ekstraksi teks dengan Ollama Vision berhasil.")
            return result
        except Exception as e:
            print(f"❌ Error saat menghubungi Ollama untuk OCR: {e}")
            return ""
    

    # --- JIKA TOGGLE MENGGUNAKAN GEMINI (FALLBACK) ---
    else:
        if not GEMINI_API_KEY or not GEMINI_MODEL:
            error_msg = "Error: GEMINI_API_KEY atau GEMINI_MODEL environment variables tidak diatur. Silakan cek file .env Anda."
            print(error_msg)
            return ""

        try:
            genai.configure(api_key=GEMINI_API_KEY)
            print(f"Menghubungi ({GEMINI_MODEL}) untuk mengekstrak teks dari gambar...")
            
            model = genai.GenerativeModel(GEMINI_MODEL)
            image_part = {
                "mime_type": mime_type,
                "data": base64_image
            }
            
            response = model.generate_content([prompt, image_part])
            
            print("Ekstraksi teks dengan Gemini Vision berhasil.")
            return response.text
        except Exception as e:
            print(f"❌ Error saat menghubungi Gemini dengan gambar: {e}")
            return ""

def is_pdf_scanned(file_path: str):
    """Detects if a PDF has no extractable text layer."""
    try:
        doc = fitz.open(file_path)
        total_text_length = sum(len(page.get_text()) for page in doc)
        return total_text_length < 100
    except Exception as e:
        print(f"Error checking PDF: {e}")
        return True

def create_searchable_pdf(text_content: str, output_path: str):
    """Creates a new, searchable PDF file from text content, with improved margin handling."""
    c = canvas.Canvas(output_path, pagesize=letter)
    width, height = letter
    left_margin = 72
    right_margin = width - 72
    top_margin = height - 72
    bottom_margin = 72
    line_height = 12
    font_size = 10

    try:
        pdfmetrics.registerFont(TTFont('DejaVuSans', 'DejaVuSans.ttf'))
        font_name = 'DejaVuSans'
    except Exception:
        print("Warning: DejaVuSans.ttf not found. Using standard font (Helvetica).")
        font_name = 'Helvetica'

    c.setFont(font_name, font_size)

    textobject = c.beginText()
    textobject.setTextOrigin(left_margin, top_margin)
    textobject.setFont(font_name, font_size)

    current_y = top_margin

    styles = getSampleStyleSheet()
    style = styles['Normal']
    style.fontName = font_name
    style.fontSize = font_size
    style.leading = line_height
    style.alignment = TA_LEFT
    style.leftIndent = 0
    style.rightIndent = 0

    paragraphs = text_content.split('\n')

    for para_text in paragraphs:
        p = Paragraph(para_text, style)
        available_width = width - left_margin - (width - right_margin)
        frame_height = p.wrap(available_width, height)[1]

        if current_y - frame_height < bottom_margin:
            c.drawText(textobject)
            c.showPage()
            textobject = c.beginText()
            textobject.setTextOrigin(left_margin, top_margin)
            textobject.setFont(font_name, font_size)
            current_y = top_margin

        current_y -= frame_height
        p.drawOn(c, left_margin, current_y)
        current_y -= line_height / 2

    c.drawText(textobject)
    c.save()
    print(f"Searchable PDF successfully created at: {output_path}")
    
# --- UPDATE FUNGSI CLASSIFY IMAGE CONTENT ---
def classify_image_content(filename: str, text_content: str) -> str:
    prompt = f"""
    Analyze the following filename and its extracted text content. Classify it into one of these categories: administrative, medicine, parking.
    - 'administrative' refers to documents like invoices, receipts, forms, letters, certificate, statement letter, or official documents.
    - 'medicine' refers to prescriptions, drug labels, medical reports, or anything related to health.
    - 'parking' refers to parking tickets, parking receipts, or signs related to parking.

    If the content does not clearly fit into any of the above categories, classify it as 'general'.

    Respond with ONLY the category name in lowercase and nothing else.

    Filename: "{filename}"
    Extracted Text: "{text_content[:1500]}..." 
    """

    print(f"🔬 Classifying content for: {filename} using [{CLASSIFICATION_PROVIDER.upper()}]")
    
    try:
        if CLASSIFICATION_PROVIDER == 'openai':
            raw_result = call_openai_vision_or_text(prompt)
        elif CLASSIFICATION_PROVIDER == 'ollama':
            raw_result = call_ollama_classification(prompt)
        else:
            # Fallback ke Gemini
            if not GEMINI_API_KEY or not GEMINI_MODEL:
                print("⚠️ Gemini API details not set. Defaulting to 'general'.")
                return "general"
            genai.configure(api_key=GEMINI_API_KEY)
            model = genai.GenerativeModel(GEMINI_MODEL)
            response = model.generate_content(prompt)
            raw_result = response.text

        # Bersihkan hasil
        category = raw_result.strip().lower()
        category = re.sub(r'[^a-z]', '', category) # Bersihkan karakter aneh
        
        # Validasi
        for valid_cat in ["administrative", "medicine", "parking"]:
            if valid_cat in category:
                print(f"✅ Classified as: {valid_cat}")
                return valid_cat
                
        print(f"⚠️ Classified as 'general' (Result was: {category})")
        return "general"
        
    except Exception as e:
        print(f"❌ Error during classification: {e}. Defaulting to 'general'.")
        return "general"

# --- UPDATE FUNGSI CLASSIFY HANDWRITTEN STATUS ---
def classify_handwritten_status(image_path: str) -> str:
    base64_image = image_to_base64(image_path)
    if not base64_image:
        return "digital"
    
    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type is None:
        mime_type = 'image/png'

    prompt = """
    You are an expert document analyst. Analyze the provided image and classify its primary content type into exactly one of two categories: 'handwritten' or 'digital'.

    Follow these strict rules for classification:
    1. Classify as 'handwritten' if the MAIN SUBSTANCE or the MAJORITY of the data in the document is written by human hand (e.g., pen/pencil on lined paper, hand-drawn tables, handwritten notes/letters).
    2. If the document is a printed template/form BUT the core data fields are filled out with handwriting, classify it as 'handwritten'.
    3. Classify as 'digital' ONLY if the document is primarily typewritten/printed using computer fonts (e.g., standard official letters, digital invoices, printed reports) and any handwriting is minimal (like just a single signature or a tiny date at the bottom).
    
    CRITICAL: Respond with EXACTLY ONE WORD. Either 'handwritten' or 'digital'. Do not add any punctuation, explanation, or extra text.
    """

    print(f"🔬 Classifying handwritten status for: {os.path.basename(image_path)} using [{CLASSIFICATION_PROVIDER.upper()}]")

    try:
        if CLASSIFICATION_PROVIDER == 'openai':
            raw_result = call_openai_vision_or_text(prompt, base64_image, mime_type)
        elif CLASSIFICATION_PROVIDER == 'ollama':
            raw_result = call_ollama_classification(prompt, base64_image)
        else:
            # Fallback ke Gemini
            if not GEMINI_API_KEY or not GEMINI_MODEL:
                print("⚠️ Gemini API details not set. Defaulting to 'digital'.")
                return "digital"
            genai.configure(api_key=GEMINI_API_KEY)
            model = genai.GenerativeModel(GEMINI_MODEL)
            image_part = {"mime_type": mime_type, "data": base64_image}
            generation_config = genai.types.GenerationConfig(temperature=0.1)
            response = model.generate_content([prompt, image_part], generation_config=generation_config)
            raw_result = response.text

        # Bersihkan dari kemungkinan karakter tak terlihat/tanda baca yang nyasar
        result = raw_result.strip().lower()
        result = re.sub(r'[^a-z]', '', result)
        
        if "handwritten" in result:
            print("✅ Classified as: handwritten")
            return "handwritten"
        else:
            print("✅ Classified as: digital")
            return "digital"
            
    except Exception as e:
        print(f"❌ Error during handwritten classification: {e}. Defaulting to 'digital'.")
        return "digital"