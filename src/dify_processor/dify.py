# src/dify_processor/dify.py

import json
import requests
from pathlib import Path

class DifyDataset:
    # Update __init__ untuk menerima parameter konfigurasi dengan default value
    def __init__(self, base_url, id, api_key, separator="=== Halaman", max_tokens=2000, chunk_overlap=300):
        self.base_url = base_url
        self.id = id
        self.api_key = api_key
        
        # Simpan konfigurasi ke dalam instance
        self.separator = separator
        self.max_tokens = int(max_tokens) # Pastikan integer
        self.chunk_overlap = int(chunk_overlap) # Pastikan integer
        
        print(f"DifyDataset Initialized (Separator: '{self.separator}', Max Tokens: {self.max_tokens}, Overlap: {self.chunk_overlap})")
        
    def upload_document_to_dataset(self, file_path):
        """Upload a document to a dataset using the API."""
        
        # Validate file exists
        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        url = f"{self.base_url}/datasets/{self.id}/document/create-by-file"
        headers = {
            'Authorization': f'Bearer {self.api_key}'
        }
        files = {
            'file': (file_path_obj.name, open(file_path, 'rb'))
        }
        
        # Gunakan variabel self.xxx di dalam payload
        data_payload = {
            "indexing_technique": "high_quality",
            "process_rule": {
                "rules": {
                    "pre_processing_rules": [
                        {"id": "remove_extra_spaces", "enabled": False},
                        {"id": "remove_urls_emails", "enabled": False}
                    ],
                    "segmentation": {
                        "separator": self.separator,       # <-- Dinamis dari ENV
                        "max_tokens": self.max_tokens,     # <-- Dinamis dari ENV
                        "chunk_overlap": self.chunk_overlap # <-- Dinamis dari ENV
                    }
                },
                "mode": "custom"
            }
        }
        
        data = {
            'data': (None, json.dumps(data_payload), 'text/plain')
        }
        
        try:
            response = requests.post(url, headers=headers, files=files, data=data)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Response status code: {e.response.status_code}")
                print(f"Response text: {e.response.text}")
            raise
        finally:
            if 'file' in files:
                files['file'][1].close()

    # --- TAMBAHKAN KODE DI BAWAH INI ---
    def get_document_id_by_name(self, document_name: str):
        """Mencari ID dokumen di Dify berdasarkan namanya."""
        url = f"{self.base_url}/datasets/{self.id}/documents"
        headers = {'Authorization': f'Bearer {self.api_key}'}
        params = {'page': 1, 'limit': 100} # Ambil hingga 100 dokumen

        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            documents = response.json().get('data', [])
            
            for doc in documents:
                if doc.get('name') == document_name:
                    print(f"Document found in Dify: {document_name} with ID: {doc.get('id')}")
                    return doc.get('id')
            
            print(f"Document not found in Dify with name: {document_name}")
            return None
        except requests.exceptions.RequestException as e:
            print(f"Failed to list documents from Dify: {e}")
            return None

    def delete_document_from_dataset(self, document_name: str):
        """Menghapus dokumen dari dataset Dify berdasarkan namanya."""
        document_id = self.get_document_id_by_name(document_name)
        if not document_id:
            return False

        url = f"{self.base_url}/datasets/{self.id}/documents/{document_id}"
        headers = {'Authorization': f'Bearer {self.api_key}'}
        
        try:
            response = requests.delete(url, headers=headers)
            response.raise_for_status()
            print(f"Successfully deleted document '{document_name}' (ID: {document_id}) from Dify.")
            return True
        except requests.exceptions.RequestException as e:
            print(f"Failed to delete document from Dify: {e}")
            return False
    # --- BATAS AKHIR PENAMBAHAN KODE ---