#!/usr/bin/env bash
set -euo pipefail

python -m mhqa.train \
  --train-csv data/Train.csv \
  --val-csv data/Val.csv \
  --subset Swa_Ken \
  --model vutuka/Llama-3.1-8B-african-aya \
  --output-dir outputs/msrh_health_qa_results \
  --adapter-dir outputs/msrh_health_qa_swa_ken
