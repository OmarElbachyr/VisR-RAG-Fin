from datasets import load_dataset, load_from_disk
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get dataset directory
dataset_path = "finetune/datasets/visual_queries_dataset_v2_0.5"
repo_id = "omarelba/visual-queries-dataset_0.5_v2"

# Load the dataset using imagefolder
print(f"Loading dataset from {dataset_path}...")

# For datasets in imagefolder format
# ds = load_dataset("imagefolder", data_dir=str(dataset_path))

# For resized datasets saved to disk
ds = load_from_disk(dataset_path)

print(f"✅ Dataset loaded!")
print(f"Train samples: {len(ds['train'])}")
print(f"Test samples: {len(ds['test'])}")

# Push dataset
print(f"Pushing dataset to {repo_id}...")
ds.push_to_hub(repo_id, private=False)
print("✅ Dataset pushed!")