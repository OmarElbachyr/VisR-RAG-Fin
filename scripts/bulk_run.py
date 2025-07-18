import subprocess

#---script for bulk run of evaluation --- define args in scripts first ---

# Define the list of (logfile, data2_file) pairs
SETUPS = [
    ("/home/laura/vqa-ir-qa/data/retrieved_pages/nomic-ai_colnomic-embed-multimodal-7b_sorted_run.json", "/home/laura/vqa-ir-qa/data/nomic_7b_molmoe_1b_cases_annotated.txt", "/home/laura/vqa-ir-qa/data/failed_nomic_7b_molmoe_1b_cases_annotated.txt"),
    ("/home/laura/vqa-ir-qa/data/retrieved_pages/vidore_colpali-v1.3_sorted_run.json", "/home/laura/vqa-ir-qa/data/colipali_molmoe_1b_cases_annotated.txt", "/home/laura/vqa-ir-qa/data/failed_colipali_molmoe_1b_cases_annotated.txt"),
    ("/home/laura/vqa-ir-qa/data/retrieved_pages/vidore_colqwen2-v1.0_sorted_run.json", "/home/laura/vqa-ir-qa/data/colqwen2_molmoe_1b_cases_annotated.txt", "/home/laura/vqa-ir-qa/data/failed_colqwen2_molmoe_1b_cases_annotated.txt"),
    ("/home/laura/vqa-ir-qa/data/retrieved_pages/vidore_colqwen2.5-v0.2_sorted_run.json", "/home/laura/vqa-ir-qa/data/colqwen25_molmoe_1b_cases_annotated.txt", "/home/laura/vqa-ir-qa/data/failed_colqwen25_molmoe_1b_cases_annotated.txt"),
    ("/home/laura/vqa-ir-qa/data/retrieved_pages/vidore_colSmol-256M_sorted_run.json", "/home/laura/vqa-ir-qa/data/colsmol_256_molmoe_1b_cases_annotated.txt", "/home/laura/vqa-ir-qa/data/failed_colsmol_256_molmoe_1b_cases_annotated.txt"),
    ("/home/laura/vqa-ir-qa/data/retrieved_pages/vidore_colSmol-500M_sorted_run.json", "/home/laura/vqa-ir-qa/data/colsmol_500_molmoe_1b_cases_annotated.txt", "/home/laura/vqa-ir-qa/data/failed_colsmol_500_molmoe_1b_cases_annotated.txt"),
    # Add more setups as needed
]

# Path to your main evaluation script
SCRIPT = "/home/laura/vqa-ir-qa/src/generators/molmoe_topX.py"

for scores, output, erroroutput in SETUPS:
    print(f"\n▶️ Running evaluation with:\n  scores: {scores}\n  output: {output}\n  errors: {erroroutput}\n")

    try:
        subprocess.run(
            ["python", SCRIPT, "--scores", scores, "--output", output, "--erroroutput", erroroutput],
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"❌ Evaluation failed for {scores}: {e}")
