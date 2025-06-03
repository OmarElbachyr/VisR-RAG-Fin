import os
import json
from pytesseract import image_to_string
from PIL import Image
from pytesseract import pytesseract

pytesseract.tesseract_cmd = "C:/Users/laura.bernardy/AppData/Local/Programs/Tesseract-OCR/tesseract.exe"
def extract_text_from_images(folder_path, output_json="all_chunks.json"):
    all_chunks = []
    image_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(".png")])

    for idx, filename in enumerate(image_files):
        file_path = os.path.join(folder_path, filename)
        print(f"Processing: {filename}")
        
        image = Image.open(file_path)
        text = image_to_string(image)

        if text.strip():  # skip empty
            all_chunks.append({
                "index": idx,
                "chunk": text.strip(),
                "filename": filename
            })

    # Save to JSON
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Done. Extracted text from {len(all_chunks)} images and saved to '{output_json}'")

# Example usage
if __name__ == "__main__":
    folder_path = "P:/vqa-benchmark/data/sampled_pages/single_page_images/BNP" 
    extract_text_from_images(folder_path)
