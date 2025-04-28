# convert pdfs to txt and preserve metadata
import os
import json
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from tqdm import tqdm

def convert_pdfs_to_text(pdf_dir="./climate_text_dataset", output_dir="./processed_texts"):
    
    os.makedirs(output_dir, exist_ok=True)
    metadata_dir = os.path.join(output_dir, "metadata")
    os.makedirs(metadata_dir, exist_ok=True)
    
    pdf_loader = DirectoryLoader(pdf_dir, glob="**/*.pdf", loader_cls=PyPDFLoader)
    pdf_documents = pdf_loader.load()
    
    print(f"Found {len(pdf_documents)} PDF documents")
    
    documents_by_source = {}
    for doc in pdf_documents:
        source = doc.metadata.get('source', '')
        if source not in documents_by_source:
            documents_by_source[source] = []
        documents_by_source[source].append(doc)
    
    for source_path, docs in tqdm(documents_by_source.items(), desc="Converting PDFs"):
        filename = os.path.basename(source_path).replace('.pdf', '')
        
        full_text = ""
        page_metadata = []
        
        for i, doc in enumerate(docs):
            full_text += f"=== Page {i+1} ===\n{doc.page_content}\n\n"
            page_metadata.append({
                "page_number": i + 1,
                "content_length": len(doc.page_content),
                "metadata": doc.metadata
            })
        
        text_output_path = os.path.join(output_dir, f"{filename}.txt")
        with open(text_output_path, 'w', encoding='utf-8') as f:
            f.write(full_text)
        
        metadata_output_path = os.path.join(metadata_dir, f"{filename}_metadata.json")
        metadata = {
            "source_file": source_path,
            "filename": filename,
            "total_pages": len(docs),
            "total_length": len(full_text),
            "pages": page_metadata
        }
        
        with open(metadata_output_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
    
    print(f"Converted {len(documents_by_source)} PDFs to text files in {output_dir}")
    print(f"Metadata saved in {metadata_dir}")

if __name__ == "__main__":
    convert_pdfs_to_text()