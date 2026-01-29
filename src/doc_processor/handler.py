# src/doc_processor/handler.py

import os
import shutil
import fitz  # PyMuPDF
from . import utils # Import our new utils module

class DocumentProcessorHandler:
    def __init__(self):
        # Define base directories
        self.input_dir = "input_files"
        self.output_dir = "output_files"
        os.makedirs(self.input_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        print("DocumentProcessorHandler Initialized")

    def process_document(self, input_path: str):
        """
        Main function to process a single document.
        This is the refactored version of your `process_document_to_searchable_pdf`.
        """
        filename = os.path.basename(input_path)
        name, extension = os.path.splitext(filename)
        output_path = os.path.join(self.output_dir, f"{name}_processed.pdf")

        print(f"\n--- Starting process for: {filename} ---")

        if extension.lower() == '.docx':
            print("Word file detected. Converting to PDF...")
            utils.convert(input_path, output_path)
            print(f"Word to PDF conversion successful: {output_path}")

        elif extension.lower() in ['.png', '.jpeg', '.jpg']:
            print("Image file detected. Using Gemma Vision...")
            final_text = utils.extract_text_with_gemini_vision(input_path)
            if final_text:
                utils.create_searchable_pdf(final_text, output_path)
            else:
                print(f"Failed to extract text from {filename}. Skipping PDF creation.")

        elif extension.lower() == '.pdf':
            if not utils.is_pdf_scanned(input_path):
                print("This PDF is already searchable. Copying file...")
                shutil.copy(input_path, output_path)
            else:
                print("Scanned PDF detected. Running Gemma Vision page by page...")
                doc = fitz.open(input_path)
                full_final_text = ""
                for i, page in enumerate(doc):
                    print(f"  - Processing page {i+1} with Gemma Vision...")
                    pix = page.get_pixmap(dpi=300)
                    img_path = f"temp_page_{i}.png"
                    pix.save(img_path)
                    page_text = utils.extract_text_with_gemini_vision(img_path)
                    full_final_text += page_text + "\n\n"
                    os.remove(img_path)
                
                if full_final_text.strip():
                    utils.create_searchable_pdf(full_final_text, output_path)
                else:
                    print(f"Failed to extract text from {filename}. Skipping PDF creation.")
        else:
            print(f"File format {extension} is not supported.")

        # Clean up the original input file
        os.remove(input_path)
        print(f"--- Finished processing: {filename} ---")