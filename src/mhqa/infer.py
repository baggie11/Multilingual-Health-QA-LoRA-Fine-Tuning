from __future__ import annotations

import argparse

import torch
from unsloth import FastLanguageModel

from .data import make_prompt


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference with base model or LoRA adapter")
    parser.add_argument("--model", default="vutuka/Llama-3.1-8B-african-aya")
    parser.add_argument("--question", required=True)
    parser.add_argument("--language", default="Swahili")
    parser.add_argument("--max-seq-length", type=int, default=1024)
    args = parser.parse_args()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )

    prompt = make_prompt(args.language, args.question)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,
        )

    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = full_response.split("### Response:")[-1].strip()
    print(answer)


if __name__ == "__main__":
    main()
