from datasets import load_dataset
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get dataset directory
dataset_path = "finetune/datasets/visual_queries_dataset"
repo_id = "omarelba/visual-queries-dataset"

# Load the dataset using imagefolder
print(f"Loading dataset from {dataset_path}...")
ds = load_dataset("imagefolder", data_dir=str(dataset_path))

print(f"✅ Dataset loaded!")
print(f"Train samples: {len(ds['train'])}")
print(f"Test samples: {len(ds['test'])}")

# Push dataset
print(f"Pushing dataset to {repo_id}...")
ds.push_to_hub(repo_id, private=False)
print("✅ Dataset pushed!")