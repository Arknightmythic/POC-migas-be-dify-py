import os
import base64
import fitz  # PyMuPDF
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

# Load environment variables from .env file
load_dotenv()

# --- PERUBAHAN DI SINI ---
# Load configuration from environment variables
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
GEMINI_MODEL = os.getenv('GEMINI_MODEL')

# --- KEMBALIKAN FUNGSI INI ---
def image_to_base64(image_path: str):
    """Membaca file gambar dan mengubahnya menjadi string Base64."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"Gagal mengubah gambar ke Base64: {e}")
        return None

# --- UBAH FUNGSI INI UNTUK MENGGUNAKAN GEMINI DENGAN BASE64 ---
def extract_text_with_gemini_vision(image_path: str):
    """Mengirim gambar (sebagai Base64) ke model Gemini untuk ekstraksi teks."""
    if not GEMINI_API_KEY or not GEMINI_MODEL:
        error_msg = "Error: GEMINI_API_KEY atau GEMINI_MODEL environment variables tidak diatur. Silakan cek file .env Anda."
        print(error_msg)
        return ""

    print(f"Mengubah {os.path.basename(image_path)} ke Base64...")
    base64_image = image_to_base64(image_path)
    if not base64_image:
        return ""
    
    # Dapatkan tipe MIME dari file gambar
    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type is None:
        print(f"Tidak dapat mendeteksi tipe MIME untuk {image_path}. Menggunakan default 'image/png'.")
        mime_type = 'image/png'

    try:
        genai.configure(api_key=GEMINI_API_KEY)
        
        print(f"Menghubungi ({GEMINI_MODEL}) untuk mengekstrak teks dari gambar...")
        
        model = genai.GenerativeModel(GEMINI_MODEL)
        
        prompt = """
                Transcribe the text from this image with high accuracy. 
                Rules:
                1. Fix broken lines: If a sentence is broken across multiple lines in the image, join them into a single continuous line.
                2. Preserve structure: Keep distinct paragraphs and numbered lists (1., 2., etc.) on separate lines.
                3. Output format: Return clean text where each bullet point or paragraph is on its own line.
                4. Do not include markdown code blocks (```).
                """

        # Buat payload konten sesuai format API Gemini
        image_part = {
            "mime_type": mime_type,
            "data": base64_image
        }
        
        response = model.generate_content([prompt, image_part])
        
        print("Ekstraksi teks dengan Gemini Vision berhasil.")
        return response.text
    except Exception as e:
        print(f"Error saat menghubungi Gemini dengan gambar: {e}")
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
    
def classify_image_content(filename: str, text_content: str) -> str:
    """
    Mengklasifikasikan konten gambar berdasarkan nama file dan teks yang diekstrak.
    Mengembalikan salah satu dari: 'administrative', 'medicine', 'parking', 'general'.
    """
    if not GEMINI_API_KEY or not GEMINI_MODEL:
        print("⚠️ Gemini API details not set. Defaulting category to 'general'.")
        return "general"

    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel(GEMINI_MODEL)
        
        prompt = f"""
        Analyze the following filename and its extracted text content. Classify it into one of these categories: administrative, medicine, parking.
        - 'administrative' refers to documents like invoices, receipts, forms, letters, certificate, statement letter, or official documents.
        - 'medicine' refers to prescriptions, drug labels, medical reports, or anything related to health.
        - 'parking' refers to parking tickets, parking receipts, or signs related to parking.

        If the content does not clearly fit into any of the above categories, classify it as 'general'.

        Respond with ONLY the category name in lowercase and nothing else.

        Filename: "{filename}"
        Extracted Text: "{text_content[:1500]}..." 
        """ # Batasi teks untuk efisiensi

        print(f"🔬 Classifying content for: {filename}")
        response = model.generate_content(prompt)
        
        # Bersihkan respons untuk memastikan hanya kategori yang dikembalikan
        category = response.text.strip().lower()
        
        # Validasi respons
        if category in ["administrative", "medicine", "parking", "general"]:
            print(f"✅ Classified as: {category}")
            return category
        else:
            print(f"⚠️ LLM returned an invalid category ('{category}'). Defaulting to 'general'.")
            return "general"
            
    except Exception as e:
        print(f"❌ Error during classification: {e}. Defaulting to 'general'.")
        return "general"