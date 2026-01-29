import os
from pathlib import Path
from langchain_experimental.text_splitter import SemanticChunker
from dotenv import load_dotenv

from .llm import LLM
from .dify import DifyDataset
from .util import Util

load_dotenv()

def process_document_with_llm(input_folder, output_folder, file_name, llm, dataset):
    """Process all PDF and TXT files in the input folder, chunk them semantically, and extract information using any OpenAI-compatible LLM."""

    util = Util()

    # Create input and output folder if it doesn't exist
    input_path = Path(input_folder)
    input_path.mkdir(exist_ok=True)
    output_path = Path(output_folder)
    output_path.mkdir(exist_ok=True)
    
    file_path = input_path / file_name
    formatted_chunks = []

    if file_path.is_file() and file_path.suffix.lower() in ['.pdf', '.txt']:
        print(f"Processing: {file_path.name}")
        
        # Read file content based on extension
        if file_path.suffix.lower() == '.pdf':
            text_content = util.read_pdf_file(file_path)
        else:  # .txt file
            text_content = util.read_txt_file(file_path)
        
        if not text_content:
            print(f"Warning: No content extracted from {file_path.name}")
            return
        
        # Split text into semantic chunks
        text_splitter = SemanticChunker(llm.embedding_client)
        try:
            docs = text_splitter.create_documents([text_content])
            print(f"Created {len(docs)} chunks from {file_path.name}")
        except Exception as e:
            print(f"Error creating chunks for {file_path.name}: {e}")
            return
        
        # Process each chunk
        for i, doc in enumerate(docs):
            chunk_text = doc.page_content
            print(f"Processing chunk {i+1}/{len(docs)} from {file_path.name}")
            
            # Extract information using LLM
            document_topic, chunk_description = llm.extract_document_info(chunk_text)
            
            # Format the chunk
            formatted_chunk = f"""document title: {file_path.stem}
document topic: {document_topic}
chunk_description: {chunk_description}

{chunk_text}"""
            
            formatted_chunks.append(formatted_chunk)

        # Save all chunks to output file
        output_file = output_path / f"{file_path.name}.txt"
        if formatted_chunks:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write('\n===[TEXT]===\n'.join(formatted_chunks))
            print(f"Saved {len(formatted_chunks)} processed chunks to {output_file}")

            try:
                dataset.upload_document_to_dataset(file_path=output_file)
                print(f"Uploaded document {file_path.name} to the dataset!")
            except Exception as e:
                print(f"Upload document {file_path.name} failed: {e}")
        else:
            print("No chunks were processed.")

if __name__ == "__main__":
    llm = LLM(
        llm_base_url=os.getenv("LLM_BASE_URL"),
        llm_model=os.getenv("LLM_MODEL"),
        llm_api_key=os.getenv("LLM_API_KEY"),
        embedding_base_url=os.getenv("EMBEDDINGS_BASE_URL"),
        embedding_model=os.getenv("EMBEDDINGS_MODEL"),
    )

    dataset = DifyDataset(
        base_url=os.getenv("DATASET_BASE_URL"),
        id=os.getenv("DATASET_ID"),
        api_key=os.getenv("DATASET_API_KEY")
    )

    input_path = Path("input")
    input_path.mkdir(exist_ok=True)
    
    # Process all files in the input folder
    for file_path in input_path.glob('*'):
        try:
            process_document_with_llm(
                input_folder="input",
                output_folder="output",
                file_name=file_path.name,
                llm=llm,
                dataset=dataset
            )
        except Exception as e:
            print(f"Error: {e}")
