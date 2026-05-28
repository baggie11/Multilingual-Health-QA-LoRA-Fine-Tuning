# Multilingual Health QA (LoRA Fine-Tuning)

A reproducible training pipeline for multilingual health question answering using Unsloth, 4-bit quantization, and LoRA adapters.

## Overview

Healthcare QA quality is often weaker in low-resource languages. This project fine-tunes an African-language base model on language-specific subsets (for example `Swa_Ken`) using supervised instruction tuning from `(input -> output)` examples.

## Key Features

- Subset-specific training via `subset` filtering
- Alpaca-style instruction formatting for SFT
- Parameter-efficient LoRA fine-tuning
- 4-bit model loading for lower GPU memory usage
- Step-wise validation with best-checkpoint selection by `eval_loss`

## Approach

1. Load `Train.csv` and `Val.csv`
2. Filter rows by target `subset`
3. Format each sample into instruction/input/response prompt text
4. Build Hugging Face datasets
5. Load base model in 4-bit
6. Attach LoRA adapters to attention/MLP projection layers
7. Train with `trl.SFTTrainer`
8. Save LoRA adapter and tokenizer

## Default Training Configuration

- Base model: `vutuka/Llama-3.1-8B-african-aya`
- Sequence length: `1024`
- LoRA rank/alpha/dropout: `16 / 16 / 0.0`
- Epochs: `2`
- Per-device train batch size: `2`
- Gradient accumulation: `4` (effective batch size `8`)
- Learning rate: `2e-4`
- Optimizer: `adamw_8bit`
- Scheduler: `cosine`

## Repository Structure

- `src/mhqa/train.py`: training CLI
- `src/mhqa/infer.py`: inference CLI
- `src/mhqa/data.py`: data loading and prompt formatting
- `src/mhqa/config.py`: default training config and language map
- `scripts/train_swahili.sh`: example training command
- `configs/train_swahili.example.yaml`: sample settings file
- `data/`: dataset location (`Train.csv`, `Val.csv`)

## Setup

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
# .venv\Scripts\Activate.ps1

pip install --upgrade pip
pip install -e .
```

Colab-compatible dependency pinning used in experiments:

```bash
pip install --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes
```

## Dataset Requirements

Expected CSV columns:

- `input`: health question
- `output`: reference answer
- `subset`: language/country code (example: `Swa_Ken`)

Place files in:

- `data/Train.csv`
- `data/Val.csv`

## Training

```bash
python -m mhqa.train \
  --train-csv data/Train.csv \
  --val-csv data/Val.csv \
  --subset Swa_Ken \
  --model vutuka/Llama-3.1-8B-african-aya \
  --output-dir outputs/msrh_health_qa_results \
  --adapter-dir outputs/msrh_health_qa_swa_ken
```

Runtime diagnostics printed during training include:

- train/validation sample counts
- average and 95th-percentile text length
- token length of first formatted sample

## Inference

```bash
python -m mhqa.infer \
  --model outputs/msrh_health_qa_swa_ken \
  --question "Je, ARVs huponya au kutibu Ukimwi?" \
  --language Swahili
```

## Outputs

- Checkpoints/logs: `outputs/msrh_health_qa_results`
- Final adapter/tokenizer: `outputs/msrh_health_qa_swa_ken`

## Practical Notes

- This workflow was developed and validated on Google Colab free GPU using memory-efficient settings (4-bit + LoRA).
- It can be extended to other subsets (for example `Aka_Gha`, `Lug_Uga`, `Amh_Eth`, and English subsets) by changing `--subset`.
- A strong future direction is a Mixture-of-Experts (MoE) architecture with language/topic-specialized experts.

## Limitations and Safety

- This is a research pipeline, not a clinical diagnosis system.
- Outputs may be incorrect or unsafe; human/clinical review is required before real-world use.
