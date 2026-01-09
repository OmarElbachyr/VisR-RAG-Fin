import argparse
import os
import shutil
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import TrainingArguments
import wandb

from colpali_engine.data.dataset import ColPaliEngineDataset
from colpali_engine.loss.late_interaction_losses import ColbertLoss, ColbertPairwiseCELoss
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
from colpali_engine.trainer.colmodel_torch_training import ColModelTorchTraining
from colpali_engine.trainer.colmodel_training import ColModelTraining, ColModelTrainingConfig
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def load_train_set(dataset_name: str = "SnT-TruX/all-wiki-100dpi") -> ColPaliEngineDataset:
    dataset = load_dataset(dataset_name, split="train")

    train_dataset = ColPaliEngineDataset(dataset, pos_target_column_name="image")

    return train_dataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tau", type=float, default=0.02, help="temperature for loss function")
    p.add_argument("--trainer", type=str, default="hf", choices=["torch", "hf"], help="trainer to use")
    p.add_argument("--loss", type=str, default="ce", choices=["ce", "pairwise"], help="loss function to use")
    return p.parse_args()


class Config:
    """Configuration for the training run."""
    model_name = "vidore/colqwen2.5-v0.2"
    dataset_name = "omarelba/visual-queries-dataset"
    output_dir = "finetune/checkpoints/colqwen2.5-v0.2-visual-queries_5e-5" # where to write model + script copy

    # Training parameters
    num_train_epochs = 1
    gradient_accumulation_steps = 4
    per_device_train_batch_size = 4
    per_device_eval_batch_size = 4
    gradient_checkpointing = True
    gradient_checkpointing_kwargs = {"use_reentrant": False}
    eval_strategy = "steps"
    dataloader_num_workers = 8
    save_steps = 100
    logging_steps = 10
    eval_steps = 50
    warmup_steps = 100
    save_total_limit = 10
    learning_rate = 5e-5
    
    # misc: for logging and wandb
    wandb_project = "fin-ir"
    wandb_experiment_name = "colqwen2.5-v0.2-visual-queries-TruX"
    num_gpus = torch.cuda.device_count()
    effective_batch = per_device_train_batch_size * gradient_accumulation_steps * num_gpus
    precision = "bfloat16"
    
    # LoRa parameters if using PEFT
    use_peft = True # set to False to disable PEFT
    lora_r = 32
    lora_alpha = 32
    lora_dropout = 0.1
    lora_init_lora_weights = "gaussian"
    lora_bias = "none"
    lora_task_type = "FEATURE_EXTRACTION"
    lora_target_modules = r"^.*(?:layer|block).*?(?:q_proj|k_proj|v_proj|o_proj|down_proj|gate_proj|up_proj)$"



if __name__ == "__main__":
    args = parse_args()

    if args.loss == "ce":
        loss_func = ColbertLoss(
            temperature=args.tau,
            normalize_scores=True,
            use_smooth_max=False,
            pos_aware_negative_filtering=False,
        )
    elif args.loss == "pairwise":
        loss_func = ColbertPairwiseCELoss(
            normalize_scores=False,
        )
    else:
        raise ValueError(f"Unknown loss function: {args.loss}")

    config = ColModelTrainingConfig(
        output_dir=Config.output_dir,
        processor=ColQwen2_5_Processor.from_pretrained(
            pretrained_model_name_or_path=Config.model_name,
            max_num_visual_tokens=768,
        ),
        model=ColQwen2_5.from_pretrained(
            pretrained_model_name_or_path=Config.model_name,
            torch_dtype=torch.bfloat16,
            use_cache=False,
        ),
        train_dataset=load_train_set(dataset_name=Config.dataset_name),
        eval_dataset=ColPaliEngineDataset(
            load_dataset(Config.dataset_name, split="test"), pos_target_column_name="image"
        ),
        run_eval=True,
        loss_func=loss_func,
        tr_args=TrainingArguments(
            output_dir=None,
            overwrite_output_dir=True,
            num_train_epochs=Config.num_train_epochs,
            per_device_train_batch_size=Config.per_device_train_batch_size,
            gradient_accumulation_steps=Config.gradient_accumulation_steps,
            gradient_checkpointing=Config.gradient_checkpointing,
            gradient_checkpointing_kwargs=Config.gradient_checkpointing_kwargs,
            per_device_eval_batch_size=Config.per_device_eval_batch_size,
            eval_strategy=Config.eval_strategy,
            dataloader_num_workers=Config.dataloader_num_workers,
            save_steps=Config.save_steps,
            logging_steps=Config.logging_steps,
            eval_steps=Config.eval_steps,
            warmup_steps=Config.warmup_steps,
            learning_rate=Config.learning_rate,
            save_total_limit=Config.save_total_limit,
        ),
        peft_config=LoraConfig(
            r=Config.lora_r,
            lora_alpha=Config.lora_alpha,
            lora_dropout=Config.lora_dropout,
            init_lora_weights=Config.lora_init_lora_weights,
            bias=Config.lora_bias,
            task_type=Config.lora_task_type,
            target_modules=Config.lora_target_modules,
        )
        if Config.use_peft
        else None,
    )

    # make sure output_dir exists and copy script for provenance
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    shutil.copy(Path(__file__), Path(config.output_dir) / Path(__file__).name)

    # --- wandb setup ---
    
    # WandB login
    wandb_token = os.getenv("WANDB_API_KEY")
    if wandb_token:
        wandb.login(key=wandb_token)
        print("✓ WandB login successful")
    else:
        print("⚠️  WANDB_API_KEY not found in .env file. Skipping WandB login.")

    wandb_project = Config.wandb_project
    wandb_experiment_name = Config.wandb_experiment_name
    wandb_tags = [
        Config.model_name.split("/")[-1],
        Config.dataset_name.split("/")[-1],
        f"per-device-batch-{Config.per_device_train_batch_size}",
        f"grad-accum-{Config.gradient_accumulation_steps}",
        f"effective-batch-{Config.effective_batch}",
        f"multi-gpu-{Config.num_gpus}",
        Config.precision,
    ]
    run = wandb.init(
        entity="mLux-rag",
        project=wandb_project,
        name=wandb_experiment_name,
        job_type="finetuning",
        tags=wandb_tags,
        config={
            "model_name": Config.model_name,
            "dataset_name": Config.dataset_name,
            "num_gpus": Config.num_gpus,
            "effective_batch_size": Config.effective_batch,
            "precision": Config.precision,
        },
    )
    print(f"✓ WandB run initialized: {wandb_experiment_name}")

    # --- end wandb setup ---

    trainer = ColModelTraining(config) if args.trainer == "hf" else ColModelTorchTraining(config)
    trainer.train()
    trainer.save()
