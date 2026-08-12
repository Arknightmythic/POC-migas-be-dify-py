import os
import base64
# [B-24] Nama modul `fitz` sudah deprecated ("The `fitz` API is deprecated and
# will be removed in future. Use `import pymupdf` instead"). Alias `as fitz`
# dipertahankan supaya pemakaian di bawah tidak perlu diubah.
import pymupdf as fitz
import re
import requests
# [B-14] `google.generativeai` sudah End of Life -> pakai `google-genai`.
from google import genai
from google.genai import types

# [B-50] vision_provider.py kini jadi SATU-SATUNYA lapisan provider (lokal,
# cloud, Gemini) dan dipakai bersama oleh doc_processor & image_processor.
# Letaknya masih di image_processor/ karena di situlah ia lahir; memindahkannya
# ke src/ akan memutus import fallback CLI di kedua pipeline, jadi ditahan dulu.
from ..image_processor.vision_provider import vision_chat
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.enums import TA_LEFT
from reportlab.platypus import Paragraph
from reportlab.lib.styles import getSampleStyleSheet
# [B-20] `from docx2pdf import convert` DIHAPUS: tidak pernah dipakai di file
# ini (satu-satunya convert() yang dipanggil adalah method milik
# HtmlToDocx di src/image_processor/html_to_docx.py). Import itu berbahaya
# karena docx2pdf menarik pywin32 dan mensyaratkan Microsoft Word terinstall,
# sehingga di mesin tanpa Word seluruh modul ini gagal di-import -- padahal
# fungsinya tidak dibutuhkan sama sekali.
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
    
def call_gemini_vision_or_text(
    prompt: str,
    base64_image: str = None,
    mime_type: str = "image/png",
    temperature: float = None,
) -> str:
    """
    Panggil Gemini untuk teks atau teks+gambar.

    [B-14] Satu helper untuk menggantikan tiga blok Gemini yang sebelumnya
    di-copy-paste di file ini, sekaligus migrasi dari paket `google.generativeai`
    (End of Life) ke `google-genai`.

    PERBEDAAN PENTING antar SDK: SDK lama menerima gambar sebagai
    {"mime_type": ..., "data": <string base64>}, sedangkan SDK baru
    (types.Part.from_bytes) menerima BYTES mentah. Jadi base64 harus di-decode
    dulu -- kalau tidak, Gemini menerima sampah dan OCR-nya kacau tanpa error.
    """
    if not GEMINI_API_KEY or not GEMINI_MODEL:
        print("⚠️ GEMINI_API_KEY atau GEMINI_MODEL belum di-set di .env")
        return ""

    client = genai.Client(api_key=GEMINI_API_KEY)

    contents = [types.Part.from_text(text=prompt)]
    if base64_image:
        try:
            raw = base64.b64decode(base64_image)
        except Exception as e:
            print(f"❌ Gambar base64 tidak valid: {e}")
            return ""
        contents.append(types.Part.from_bytes(data=raw, mime_type=mime_type))

    cfg = {}
    if temperature is not None:
        cfg["temperature"] = temperature

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=contents,
        config=types.GenerateContentConfig(**cfg) if cfg else None,
    )
    return response.text or ""


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
    """
    Helper untuk memanggil Ollama (Text atau Vision) sesuai CLASSIFICATION_PROVIDER.

    [B-50] Dulu fungsi ini punya implementasi HTTP SENDIRI, terpisah dari
    vision_provider.py. Duplikasi itu sumber bug yang berulang:
      - B-06: URL di-hardcode di satu jalur sementara jalur lain baca .env
      - num_ctx (B-46) hanya diperbaiki di satu jalur
      - retry, log token, dan penanganan truncation cuma ada di satu jalur
    Sekarang mendelegasikan ke vision_chat() supaya SATU implementasi provider
    dipakai bersama. CLASSIFICATION_PROVIDER dan VISION_PROVIDER tetap dua
    saklar yang independen -- yang disatukan implementasinya, bukan pilihannya.

    Efek sampingnya: CLASSIFICATION_PROVIDER otomatis ikut mendukung
    'ollama-cloud' tanpa menduplikasi logika auth/model cloud di sini.
    """
    # '/no_think' adalah konvensi Qwen2/3 untuk menekan tag <think>.
    # Untuk cloud (gemma4) pengendalinya parameter `think`, dan vision_provider
    # membuang penanda ini pada jalur Gemini. Tetap dikirim supaya perilaku
    # Ollama lokal (qwen2.5vl) tidak berubah.
    if "/no_think" not in prompt:
        prompt = f"/no_think\n{prompt}"

    messages = [{"role": "user", "content": prompt}]
    if base64_image:
        messages[0]["images"] = [base64_image]

    # Plafon token harus LONGGAR. Fungsi ini melayani dua hal sekaligus:
    # klasifikasi (jawabannya 1-3 token) DAN transkripsi OCR satu halaman penuh
    # (bisa ribuan token). Versi lama tidak mengirim num_predict sama sekali,
    # artinya tanpa batas -- jadi memasang batas ketat di sini akan MEMOTONG
    # hasil OCR pada dokumen padat, gejalanya persis B-46/B-48.
    # Plafon hanya batas atas; penagihan/komputasi mengikuti pemakaian nyata.
    try:
        return vision_chat(
            messages,
            max_tokens=int(os.getenv("CLASSIFICATION_MAX_TOKENS", "8192")),
            temperature=0.1,   # rendah agar hasilnya deterministik
            provider=CLASSIFICATION_PROVIDER,
        ).strip()
    except Exception as e:
        print(f"❌ Error saat menghubungi {CLASSIFICATION_PROVIDER}: {e}")
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
    
    elif CLASSIFICATION_PROVIDER in ('ollama', 'ollama-cloud'):
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
            print(f"Menghubungi ({GEMINI_MODEL}) untuk mengekstrak teks dari gambar...")
            result = call_gemini_vision_or_text(prompt, base64_image, mime_type)
            print("Ekstraksi teks dengan Gemini Vision berhasil.")
            return result
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
        elif CLASSIFICATION_PROVIDER in ('ollama', 'ollama-cloud'):
            raw_result = call_ollama_classification(prompt)
        else:
            # Fallback ke Gemini
            if not GEMINI_API_KEY or not GEMINI_MODEL:
                print("⚠️ Gemini API details not set. Defaulting to 'general'.")
                return "general"
            raw_result = call_gemini_vision_or_text(prompt)

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
        elif CLASSIFICATION_PROVIDER in ('ollama', 'ollama-cloud'):
            raw_result = call_ollama_classification(prompt, base64_image)
        else:
            # Fallback ke Gemini
            if not GEMINI_API_KEY or not GEMINI_MODEL:
                print("⚠️ Gemini API details not set. Defaulting to 'digital'.")
                return "digital"
            raw_result = call_gemini_vision_or_text(
                prompt, base64_image, mime_type, temperature=0.1
            )

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