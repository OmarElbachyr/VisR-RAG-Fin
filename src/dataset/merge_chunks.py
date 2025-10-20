import pandas as pd
from pathlib import Path

QA_CSV = Path("src/dataset/chunks/chunked_pages_category_A.csv")
NOISE_CSV = Path("src/dataset/chunks/chunked_noise_pages_test.csv")
OUTPUT_CSV = Path("src/dataset/chunks/chunked_all_pages.csv")

# Read both CSVs
df_qa = pd.read_csv(QA_CSV)
df_noise = pd.read_csv(NOISE_CSV)

# Merge
df_merged = pd.concat([df_qa, df_noise], ignore_index=True)

# Save
df_merged.to_csv(OUTPUT_CSV, index=False)

print(f"✅ Merged: {len(df_qa)} QA rows + {len(df_noise)} noise rows = {len(df_merged)} total rows")
print(f"Saved to: {OUTPUT_CSV}")