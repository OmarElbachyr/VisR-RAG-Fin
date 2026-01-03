import os
import json
import fitz
import layoutparser as lp
import numpy as np
from PIL import Image
from PIL import ImageFont
from datetime import datetime

# PIL patches
if not hasattr(Image, 'LINEAR'):
    Image.LINEAR = Image.Resampling.BILINEAR

if not hasattr(ImageFont.FreeTypeFont, 'getsize'):
    def getsize(font, text):
        bbox = font.getbbox(text)
        return (bbox[2] - bbox[0], bbox[3] - bbox[1])
    ImageFont.FreeTypeFont.getsize = getsize

# GPU-enabled LayoutParser model
model = lp.Detectron2LayoutModel(
    config_path="lp://PubLayNet/mask_rcnn_X_101_32x8d_FPN_3x/config",
    extra_config=[
        "MODEL.DEVICE", "cuda",
        "MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.5
    ],
    label_map={0:"Text",1:"Title",2:"List",3:"Table",4:"Figure"}
)

def is_visual_page_from_pix(pix):
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    image_np = np.array(img)
    layout = model.detect(image_np)
    return any(block.type == "Figure" for block in layout)

def process_pdf(pdf_path):
    data = {
        "filename": os.path.basename(pdf_path),
        "total_pages": 0,
        "visual_pages": 0,
        "non_visual_pages": 0,
        "visual_page_numbers": [],
        "processed_at": datetime.utcnow().isoformat() + "Z"
    }
    try:
        doc = fitz.open(pdf_path)
        total_pages = doc.page_count
        data["total_pages"] = total_pages

        for i in range(total_pages):
            page = doc.load_page(i)
            mat = fitz.Matrix(2, 2)
            pix = page.get_pixmap(matrix=mat)
            visual = is_visual_page_from_pix(pix)

            if visual:
                data["visual_pages"] += 1
                data["visual_page_numbers"].append(i + 1)
            else:
                data["non_visual_pages"] += 1

        doc.close()

    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")

    return data

if __name__ == "__main__":
    folder = "finetune/training_documents"
    out_file = "finetune/visual_metadata.json"
    all_metadata = []

    for filename in os.listdir(folder):
        if filename.lower().endswith(".pdf"):
            path = os.path.join(folder, filename)
            result = process_pdf(path)

            # Only keep entry if there *are* visual pages
            if result["visual_pages"] > 0:
                all_metadata.append(result)
                print(f"Included {filename} → {result['visual_pages']} visual pages")
            else:
                print(f"Skipped {filename} → no visual pages")

    # Save JSON file
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_metadata, f, indent=2)

    print(f"\nSaved metadata for {len(all_metadata)} PDFs → {out_file}")
