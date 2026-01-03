from pydantic import BaseModel
from typing import Tuple
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image
import io
import json
import fitz  # PyMuPDF

# Load environment variables from .env file
load_dotenv()


class RetrievalQuery(BaseModel):
    broad_topical_query: str
    broad_topical_explanation: str
    specific_detail_query: str
    specific_detail_explanation: str
    visual_element_query: str
    visual_element_explanation: str


def get_retrieval_prompt(prompt_path: str) -> str:
    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt = f.read()
    return prompt


# Load API key from environment
api_key = os.getenv("GOOGLE_API_KEY")
# Create client, using API key
client = genai.Client(api_key=api_key)

def generate_response(prompt: str, image, model_name) -> str:
    """
    Generate a Gemini text response for a prompt + image.
    
    `image` can be:
      • PIL.Image.Image
      • raw bytes (e.g., from fitz.Pixmap.tobytes())
    """

    # --- Convert fitz / PIL to a single Part ---
    if isinstance(image, Image.Image):
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        buffer.seek(0)
        data_bytes = buffer.read()
        mime_type = "image/jpeg"
    elif isinstance(image, (bytes, bytearray)):
        data_bytes = image
        mime_type = "image/jpeg"
    else:
        raise TypeError("Image must be bytes or PIL.Image")

    image_part = types.Part.from_bytes(
        data=data_bytes, 
        mime_type=mime_type
    )

    # --- Call Gemini ---
    response = client.models.generate_content(
        model=model_name,  # latest multimodal
        contents=[prompt, image_part],
        config=types.GenerateContentConfig(
        ),
    )

    return response.text


if __name__ == "__main__":
    # Configuration
    limit = None  # Set to None to process all PDFs, or a number to limit (e.g., 5)
    model_name = "gemini-3-flash-preview"
    prompt_path = "finetune/scripts/prompt.txt"

    # Load visual metadata
    metadata_path = "finetune/visual_metadata.json"
    training_docs_path = "finetune/training_documents"
    
    # Load prompt
    prompt = get_retrieval_prompt(prompt_path)
    # print(f"Prompt for '{prompt_path}':")
    # print(prompt)

    if not os.path.exists(metadata_path):
        print(f"Metadata file not found: {metadata_path}")
        exit(1)
    
    with open(metadata_path, "r", encoding="utf-8") as f:
        visual_metadata = json.load(f)
    
    print(f"Loaded {len(visual_metadata)} documents from metadata")
    if limit is not None:
        print(f"Processing limit: {limit} PDFs")
    
    # Count PDFs with visual pages
    total_visual_pdfs = sum(1 for doc in visual_metadata if doc.get("visual_page_numbers"))
    
    # Store all results
    results = []
    processed_pdfs = 0
    pdf_index = 0
    
    # Process visual pages only
    for doc_info in visual_metadata:
        filename = doc_info.get("filename")
        visual_pages = doc_info.get("visual_page_numbers", [])
        
        if not visual_pages:
            continue
        
        # Check if we've reached the limit
        if limit is not None and processed_pdfs >= limit:
            print(f"\n⏹️  Reached processing limit of {limit} PDFs")
            break
        
        pdf_index += 1
        
        # Find PDF in training_documents
        pdf_path = os.path.join(training_docs_path, filename)
        
        if not os.path.exists(pdf_path):
            print(f"⚠️  [{pdf_index}] PDF not found: {pdf_path}")
            continue
        
        total_pdfs = limit if limit is not None else total_visual_pdfs
        print(f"\n📄 [{pdf_index}/{total_pdfs}] Processing: {filename} ({len(visual_pages)} visual pages)")
        
        try:
            # Open PDF
            pdf_document = fitz.open(pdf_path)
            
            for page_num in visual_pages:
                # PDF pages are 0-indexed, but visual_page_numbers are 1-indexed
                page_index = page_num - 1
                
                if page_index < 0 or page_index >= len(pdf_document):
                    print(f"   ⚠️  Page {page_num} out of range (PDF has {len(pdf_document)} pages)")
                    continue
                
                print(f"   Processing page {page_num}...", end=" ", flush=True)
                
                # Render page to image
                page = pdf_document[page_index]
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for better quality
                image_data = pix.tobytes("ppm")
                
                # Convert to PIL Image
                from io import BytesIO
                img = Image.open(BytesIO(pix.tobytes("png")))
                
                # Generate response
                try:
                    response = generate_response(prompt, img, model_name)
                    print("✓")
                    
                    # Parse JSON from markdown code blocks
                    parsed_response = json.loads(
                        response.split("```json")[1].split("```")[0]
                    )
                    
                    # Save result
                    results.append({
                        "filename": filename,
                        "page_number": page_num,
                        "response": parsed_response,
                        "total_pages": doc_info.get("total_pages"),
                        "visual_pages": doc_info.get("visual_pages")
                    })
                except Exception as e:
                    print(f"✗ Error: {str(e)[:100]}")
            
            pdf_document.close()
            processed_pdfs += 1
            
        except Exception as e:
            print(f"   ✗ Error processing PDF: {str(e)}")
    
    # Save all results to JSON
    output_path = "finetune/scripts/visual_pages_responses.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Processing complete!")
    print(f"   PDFs processed: {processed_pdfs}")
    print(f"   Total responses generated: {len(results)}")
    print(f"   Results saved to: {output_path}")

