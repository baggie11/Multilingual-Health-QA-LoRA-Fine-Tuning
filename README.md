# Multilingual Health QA (LoRA Fine-Tuning)

This repository trains a multilingual health question-answering model using parameter-efficient fine-tuning (LoRA) on top of open base LLMs (via Unsloth).

## Problem and Goal

Healthcare Q&A systems are often weak in low-resource languages. This project adapts an African-language base model to answer health questions in target subsets (for example `Swa_Ken`) by fine-tuning on `(input -> output)` supervision.

## Approach Used

1. Subset-focused training: data is filtered by `subset` (example `Swa_Ken`) to specialize language behavior.
2. Instruction tuning: every sample is converted to an Alpaca-style prompt (`Instruction`, `Input`, `Response`).
3. LoRA fine-tuning: adapters are trained on attention + MLP projections only.
4. 4-bit loading: base model is quantized for practical GPU memory usage.
5. Step-wise validation: best checkpoint is selected using `eval_loss`.

## Model and Training Defaults (Updated)

- Base model: `vutuka/Llama-3.1-8B-african-aya`
- Sequence length: `1024`
- LoRA rank: `16`
- LoRA alpha: `16`
- LoRA dropout: `0.0`
- Epochs: `2`
- Per-device train batch size: `2`
- Gradient accumulation: `4`
- Effective batch size: `8`
- Learning rate: `2e-4`
- Optimizer: `adamw_8bit`
- Scheduler: `cosine`

## Repository Structure

- `src/mhqa/train.py`: training CLI and end-to-end training run
- `src/mhqa/infer.py`: inference CLI for quick question answering
- `src/mhqa/data.py`: data loading, filtering, and prompt formatting
- `src/mhqa/config.py`: train defaults and language mapping
- `scripts/train_swahili.sh`: convenience training command
- `configs/train_swahili.example.yaml`: sample experiment settings
- `data/`: place `Train.csv` and `Val.csv`

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

Colab compatibility install (matching your latest stack):

```bash
pip install --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes
```

## Dataset Format

Expected CSV columns:

- `input`: health question
- `output`: expected answer
- `subset`: language/country code (example `Swa_Ken`)

Place files at:

- `data/Train.csv`
- `data/Val.csv`

## Train

```bash
python -m mhqa.train \
  --train-csv data/Train.csv \
  --val-csv data/Val.csv \
  --subset Swa_Ken \
  --model vutuka/Llama-3.1-8B-african-aya \
  --output-dir outputs/msrh_health_qa_results \
  --adapter-dir outputs/msrh_health_qa_swa_ken
```

The training script also prints:

- sample counts for train/val
- average and 95th percentile character lengths
- token length of first formatted sample

## Inference

```bash
python -m mhqa.infer \
  --model outputs/msrh_health_qa_swa_ken \
  --question "Je, ARVs huponya au kutibu Ukimwi?" \
  --language Swahili
```

## Outputs

- Training artifacts/checkpoints: `outputs/msrh_health_qa_results`
- Final LoRA adapter + tokenizer: `outputs/msrh_health_qa_swa_ken`

## Limitations and Safety Notes

- This is a research pipeline, not a clinical diagnosis system.
- Model answers can still be wrong or unsafe; add human/clinical review before deployment.

## Practical Notes

- This workflow was developed and validated on Google Colab free GPU settings using memory-efficient training choices (4-bit loading + LoRA).
- The same pipeline can be easily extended to other dataset subsets (for example `Aka_Gha`, `Lug_Uga`, `Amh_Eth`, and English subsets) by changing `--subset`.
- A strong next-step improvement is a Mixture-of-Experts (MoE) style architecture, where experts can specialize by language or medical topic while sharing a common backbone.

## Push to GitHub

```bash
git init
git add .
git commit -m "Initial multilingual health QA training pipeline"
git branch -M main
git remote add origin <YOUR_GITHUB_REPO_URL>
git push -u origin main
```
