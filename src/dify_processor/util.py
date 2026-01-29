import PyPDF2
import re

class Util:
    def __init__(self):
        print("Util Initialized")

    def clean_text(self, text):
        """Clean extracted text by removing extra whitespace and special characters."""

        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s.,!?;:()\-"]', '', text)
        text = re.sub(r'[.,!?;:]+', lambda m: m.group(0)[0], text)
        return text.strip()

    def read_pdf_file(self, file_path):
        """Extract text from PDF file."""

        text = ""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
        except Exception as e:
            print(f"Error reading PDF {file_path}: {e}")
            return ""
        
        return self.clean_text(text)

    def read_txt_file(self, file_path):
        """Read text from TXT file."""

        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
        except Exception as e:
            print(f"Error reading TXT {file_path}: {e}")
            return ""
        
        return self.clean_text(text)
