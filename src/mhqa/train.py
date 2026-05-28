from __future__ import annotations

import argparse
import gc
import os

import torch
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel

from .config import TrainConfig
from .data import load_and_filter, to_hf_dataset


def run_training(cfg: TrainConfig) -> None:
    os.environ["WANDB_DISABLED"] = "true"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    torch.cuda.empty_cache()
    gc.collect()

    train_df, val_df = load_and_filter(str(cfg.train_csv), str(cfg.val_csv), cfg.subset_code)
    print(f"Training samples ({cfg.subset_code}): {len(train_df)}")
    print(f"Validation samples ({cfg.subset_code}): {len(val_df)}")
    train_dataset = to_hf_dataset(train_df, cfg.language_name)
    val_dataset = to_hf_dataset(val_df, cfg.language_name)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.model_name,
        max_seq_length=cfg.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    print(f"Loaded model: {cfg.model_name}")

    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg.lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=cfg.seed,
    )
    print(f"Trainable parameters: {model.num_parameters(only_trainable=True):,}")

    train_lengths = train_df["input"].astype(str).str.len() + train_df["output"].astype(str).str.len()
    print(f"Average input+output length: {train_lengths.mean():.0f} chars")
    print(f"95th percentile length: {train_lengths.quantile(0.95):.0f} chars")
    sample_tokens = tokenizer.encode(train_dataset[0]["text"])
    print(f"Tokenized length of first sample: {len(sample_tokens)} tokens")

    training_args = TrainingArguments(
        output_dir=str(cfg.output_dir),
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        warmup_ratio=cfg.warmup_ratio,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        logging_steps=cfg.logging_steps,
        eval_strategy="steps",
        eval_steps=cfg.eval_steps,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        report_to="none",
        optim="adamw_8bit",
        lr_scheduler_type="cosine",
        seed=cfg.seed,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        dataset_text_field="text",
        max_seq_length=cfg.max_seq_length,
        args=training_args,
    )

    print("Starting training...")
    trainer.train()
    cfg.adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(cfg.adapter_dir))
    tokenizer.save_pretrained(str(cfg.adapter_dir))


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train multilingual health QA LoRA adapter")
    p.add_argument("--train-csv", default="data/Train.csv")
    p.add_argument("--val-csv", default="data/Val.csv")
    p.add_argument("--subset", default="Swa_Ken", help="Dataset subset code, e.g. Swa_Ken")
    p.add_argument("--model", default="vutuka/Llama-3.1-8B-african-aya")
    p.add_argument("--output-dir", default="outputs/msrh_health_qa_results")
    p.add_argument("--adapter-dir", default="outputs/msrh_health_qa_swa_ken")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    cfg = TrainConfig(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        subset_code=args.subset,
        model_name=args.model,
        output_dir=args.output_dir,
        adapter_dir=args.adapter_dir,
    )
    run_training(cfg)


if __name__ == "__main__":
    main()
