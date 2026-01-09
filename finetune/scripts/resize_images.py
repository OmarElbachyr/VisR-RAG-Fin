
from functools import partial
import io
from datasets import load_dataset, DatasetDict
from PIL import Image


image_resize_ratio = 0.5
dataset_name = 'finetune/datasets/visual_queries_dataset'
output_dataset_name = f'finetune/datasets/visual_queries_dataset_{image_resize_ratio}'

# Load both train and test splits
dataset = load_dataset(dataset_name)

print("Original dataset:")
print(f"Train: {len(dataset['train'])} samples")
print(f"Test: {len(dataset['test'])} samples")

# Show some examples of original sizes
print("\nOriginal image sizes (examples):")
for i in range(min(3, len(dataset['train']))):
    img = dataset['train'][i]['image']
    print(f"  Train sample {i}: {img.size[0]}x{img.size[1]} pixels")

def resize_image(example, ratio: float = 1.0, jpeg_quality: int = 90):
    """Resize image by ratio (1.0 = original, 0.5 = 50%, etc.) and encode as JPEG."""
    img = example["image"]
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)

    if ratio != 1.0:
        w, h = img.size
        new_w = int(w * ratio)
        new_h = int(h * ratio)
        img = img.resize((new_w, new_h), resample=Image.BILINEAR)
    
    # Convert to RGB if needed (for JPEG compatibility)
    if img.mode not in ('RGB', 'L'):
        img = img.convert('RGB')
    
    # Encode as JPEG bytes to maintain compression when saved
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG', quality=jpeg_quality, optimize=True)
    buffer.seek(0)
    
    # Load back from JPEG bytes - this ensures it's stored as JPEG in Arrow
    img = Image.open(buffer)
    
    example["image"] = img
    return example


print(f"\nResizing images by ratio: {image_resize_ratio}")
print("Encoding as JPEG with quality=90 for optimal compression")
resize_fn = partial(resize_image, ratio=image_resize_ratio, jpeg_quality=90)

# Resize both splits
resized_dataset = DatasetDict({
    'train': dataset['train'].map(resize_fn),
    'test': dataset['test'].map(resize_fn)
})

print(f"\nResized dataset:")
print(f"Train: {len(resized_dataset['train'])} samples")
print(f"Test: {len(resized_dataset['test'])} samples")

# Show some examples of resized dimensions
print("\nResized image sizes (examples):")
for i in range(min(3, len(resized_dataset['train']))):
    img = resized_dataset['train'][i]['image']
    print(f"  Train sample {i}: {img.size[0]}x{img.size[1]} pixels")

# Save the resized dataset locally
print(f"\nSaving resized dataset to: {output_dataset_name}")
resized_dataset.save_to_disk(output_dataset_name)
print("Done!")

