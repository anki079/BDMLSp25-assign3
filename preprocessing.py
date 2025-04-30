import os
import json
import signal
import fitz  # PyMuPDF
import numpy as np
from tqdm import tqdm

# timeout handler to avoid hangs on bad pdfs
class TimeoutException(Exception):
    pass

def handler(signum, frame):
    raise TimeoutException()

signal.signal(signal.SIGALRM, handler)

INPUT_DIR = "./climate_text_dataset"
OUTPUT_DIR = "./processed_texts"
METADATA_DIR = os.path.join(OUTPUT_DIR, "metadata")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(METADATA_DIR, exist_ok=True)

pdf_paths = []
for root, _, files in os.walk(INPUT_DIR):
    for fname in files:
        if fname.lower().endswith(".pdf"):
            pdf_paths.append(os.path.join(root, fname))

print(f"Found {len(pdf_paths)} PDFs under {INPUT_DIR}")

for pdf_path in tqdm(pdf_paths, desc="Converting PDFs"):
    filename = os.path.splitext(os.path.basename(pdf_path))[0]
    try:
        # start 60 s alarm for file
        signal.alarm(60)

        doc = fitz.open(pdf_path)
        pages_meta = []
        full_text_parts = []

        for i, page in enumerate(doc):
            content = page.get_text() or ""
            full_text_parts.append(f"=== Page {i+1} ===\n{content}\n")
            pages_meta.append({
                "page_number": i + 1,
                "content_length": len(content)
            })

        doc.close()
        # cancel alarm
        signal.alarm(0)

    except TimeoutException:
        print(f"  Timeout extracting {filename}. Skipping.")
        continue
    except Exception as e:
        print(f"  Error on {filename}: {e}. Skipping.")
        continue

    full_text = "\n\n".join(full_text_parts)
    txt_path = os.path.join(OUTPUT_DIR, f"{filename}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(full_text)

    metadata = {
        "source_file": pdf_path,
        "filename": filename,
        "total_pages": len(pages_meta),
        "total_length": len(full_text),
        "pages": pages_meta
    }
    meta_path = os.path.join(METADATA_DIR, f"{filename}_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

print(f"\nConverted {len(os.listdir(OUTPUT_DIR))} text files : {OUTPUT_DIR}")
print(f"Metadata JSONs in : {METADATA_DIR}")
