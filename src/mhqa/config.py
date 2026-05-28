from dataclasses import dataclass
from pathlib import Path

LANGUAGE_MAP = {
    "Eng_Uga": "English",
    "Eng_Gha": "English",
    "Eng_Eth": "English",
    "Eng_Ken": "English",
    "Aka_Gha": "Akan",
    "Lug_Uga": "Luganda",
    "Swa_Ken": "Swahili",
    "Amh_Eth": "Amharic",
}


@dataclass
class TrainConfig:
    train_csv: Path = Path("data/Train.csv")
    val_csv: Path = Path("data/Val.csv")
    subset_code: str = "Swa_Ken"
    model_name: str = "vutuka/Llama-3.1-8B-african-aya"
    max_seq_length: int = 1024
    output_dir: Path = Path("outputs/msrh_health_qa_results")
    adapter_dir: Path = Path("outputs/msrh_health_qa_swa_ken")
    num_train_epochs: int = 2
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    logging_steps: int = 50
    eval_steps: int = 200
    save_steps: int = 200
    save_total_limit: int = 2
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    seed: int = 42

    @property
    def language_name(self) -> str:
        return LANGUAGE_MAP.get(self.subset_code, self.subset_code)
