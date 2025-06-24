import torch
from PIL import Image

from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor

model = ColQwen2_5.from_pretrained(
        "Metric-AI/ColQwen2.5-3b-multilingual-v1.0",
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",  # or "mps" if on Apple Silicon
    ).eval()
processor = ColQwen2_5_Processor.from_pretrained("Metric-AI/ColQwen2.5-3b-multilingual-v1.0")

# Your inputs
images = [
    Image.new("RGB", (32, 32), color="white"),
    Image.new("RGB", (28, 28), color="black"),
]
queries = [
    "Is attention really all you need?",
    "What is the amount of bananas farmed in Salvador?",
]

# Process the inputs
batch_images = processor.process_images(images).to(model.device)
batch_queries = processor.process_queries(queries).to(model.device)

# Forward pass
with torch.no_grad():
    image_embeddings = model(**batch_images)
    query_embeddings = model(**batch_queries)

scores = processor.score_multi_vector(query_embeddings, image_embeddings)
