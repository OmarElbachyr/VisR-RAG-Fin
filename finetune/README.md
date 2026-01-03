# PDF folders
- ``finetune/documents``: classified docs but truncated -> to be deleted
- ``finetune/original_document``s: original scraed PDFs: company calassifed only -> to be deleted
- ``finetune/training_documents``: contains PDFs used for training data

# Folders
- ``finetune/datasets``: HF datasets folder

# Scripts
- all scripts in ``finetune/preprocessing_scripts`` are for stats/preparing ``finetune/training_documents`` folder -> stats are in Notion
- ``finetune/scripts/queries_generator.py``: generate queries from PDF images using Gemini 3 Flash
- ``finetune/scripts/prepare_hf_dataset.py``: Create HF dataset folder structure for a giving generated queries from the script above.