import os
import json
from dotenv import load_dotenv
from huggingface_hub import HfApi
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor

load_dotenv()

HF_API_TOKEN = os.getenv("HF_TOKEN")
if not HF_API_TOKEN:
    raise ValueError("HF_TOKEN not found in .env")


LOCAL_MODEL_PATH = "finetune/checkpoints/colnomic-embed-multimodal-3b-visual-queries-dataset_0.5_accum1_batch16_lr5e-05_epoch_5"
HF_REPO_ID = "omarelba/colnomic_3b_visual_pages_0.5_batch16_lr5e-5_epoch_5"
PRIVATE = True  # set False for a public repo


def create_readme(local_path: str, repo_id: str, base_model: str, dataset: str) -> str:
    """
    Create a README.md file with model card metadata.
    
    Args:
        local_path: Path to the local model directory
        repo_id: HuggingFace repository ID
        base_model: Base model used for fine-tuning
        dataset: Dataset used for training
    
    Returns:
        Path to the created README.md file
    """
    readme_content = f"""---
model_name: {repo_id}s
base_model: {base_model}
pipeline_tag: visual-document-retrieval
library_name: transformers
license: apache-2.0
domain: finance
language: en
dataset: {dataset}
tags:
  - multimodal
  - multimodal-retrieval
  - document-retrieval
  - fine-tuned
  - peft
  - lora
---

# {repo_id.split('/')[-1]}

Fine-tuned {base_model} model for language-visual retrieval of financial documents.

## Training Details

- **Base Model:** {base_model}
- **Dataset:** {dataset}
- **Fine-tuning Method:** LoRA/PEFT

## Usage

```python
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor

model = ColQwen2_5.from_pretrained("{repo_id}")
processor = ColQwen2_5_Processor.from_pretrained("{repo_id}")
```
"""
    
    readme_path = os.path.join(local_path, "README.md")
    with open(readme_path, "w") as f:
        f.write(readme_content)
    print(f"✔ README.md created at {readme_path}")
    return readme_path

# Load model + processor from local path
print(f"Loading local ColQwen2.5 model from {LOCAL_MODEL_PATH}")

model = ColQwen2_5.from_pretrained(LOCAL_MODEL_PATH)
processor = ColQwen2_5_Processor.from_pretrained(LOCAL_MODEL_PATH)
print("✔ Model and processor loaded.")

# Extract base model from config
BASE_MODEL = model.config._name_or_path
print(f"✔ Extracted base model: {BASE_MODEL}")

# Try to extract dataset from training script in checkpoint
training_script_path = os.path.join(LOCAL_MODEL_PATH, "finetune_colqwen2_5.py")
if os.path.exists(training_script_path):
    try:
        with open(training_script_path, "r") as f:
            content = f.read()
            # Look for dataset_name in the script
            if 'dataset_name = "' in content:
                start = content.find('dataset_name = "') + len('dataset_name = "')
                end = content.find('"', start)
                DATASET = content[start:end]
                print(f"✔ Extracted dataset from training script: {DATASET}")
    except Exception as e:
        print(f"⚠ Could not extract dataset from script: {e}, using fallback")

# Create README.md with dynamic metadata
readme_path = create_readme(LOCAL_MODEL_PATH, HF_REPO_ID, BASE_MODEL, DATASET)

# Push to Hugging Face Hub
print(f"Pushing to Hugging Face repo {HF_REPO_ID} (private={PRIVATE}) …")
api = HfApi()

# Create repo (public/private)
try:
    api.create_repo(repo_id=HF_REPO_ID, token=HF_API_TOKEN, private=PRIVATE)
    print("✔ Repo created.")
except Exception as e:
    print(f"⚠ Repo creation skipped (maybe exists already): {e}")

# Push both model and processor
model.push_to_hub(HF_REPO_ID, use_auth_token=HF_API_TOKEN)
processor.push_to_hub(HF_REPO_ID, use_auth_token=HF_API_TOKEN)

# Upload the README.md
api.upload_file(
    path_or_fileobj=readme_path,
    path_in_repo="README.md",
    repo_id=HF_REPO_ID,
    token=HF_API_TOKEN
)
print("✔ README.md uploaded.")

# Upload the Python script used for finetuning
api.upload_folder(
    folder_path=LOCAL_MODEL_PATH,
    repo_id=HF_REPO_ID,
    allow_patterns="*.py",
    use_auth_token=HF_API_TOKEN
)

print("✅ Upload complete!")
