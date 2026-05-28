from __future__ import annotations

import pandas as pd
from datasets import Dataset


def make_prompt(language_name: str, question: str, answer: str | None = None) -> str:
    prompt = (
        "Below is an instruction that describes a task. Write a response "
        "that appropriately completes the request.\n\n"
        f"### Instruction:\nAnswer the following health question in {language_name}.\n\n"
        f"### Input:\n{question}\n\n"
        "### Response:"
    )
    if answer is not None:
        prompt += f"\n{answer}"
    return prompt


def load_and_filter(train_csv: str, val_csv: str, subset_code: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)

    train_subset = train_df[train_df["subset"] == subset_code].copy()
    val_subset = val_df[val_df["subset"] == subset_code].copy()

    if train_subset.empty or val_subset.empty:
        raise ValueError(f"No rows found for subset '{subset_code}'.")

    return train_subset, val_subset


def to_hf_dataset(df: pd.DataFrame, language_name: str) -> Dataset:
    df = df.copy()
    df["text"] = df.apply(lambda r: make_prompt(language_name, r["input"], r["output"]), axis=1)
    return Dataset.from_pandas(df[["text"]])
