import os

TRAINING_ROOT = 'finetune/training_documents'


def rename_pdfs_with_truncated(root_folder):
    """Replace 'truncated.pdf' with '.py' in PDF filenames."""
    for subdir, _, files in os.walk(root_folder):
        for filename in files:
            if filename.lower().endswith('.pdf') and 'truncated.pdf' in filename:
                old_path = os.path.join(subdir, filename)
                new_filename = filename.replace('truncated.pdf', '.py')
                new_path = os.path.join(subdir, new_filename)
                os.rename(old_path, new_path)
                print(f"Renamed: {old_path} -> {new_path}")

if __name__ == "__main__":
    rename_pdfs_with_truncated(TRAINING_ROOT)