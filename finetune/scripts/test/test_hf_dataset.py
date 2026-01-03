from datasets import load_dataset
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get dataset directory
dataset_name = "omarelba/visual-queries-dataset"

# Load the dataset using imagefolder
print(f"Loading dataset from {dataset_name}...")
ds = load_dataset(dataset_name)

print(f"✅ Dataset loaded!")
print(f"Train samples: {len(ds['train'])}")
print(f"Test samples: {len(ds['test'])}")