# train_roberta_mlm.py

import wandb
import huggingface_hub
import logging
import torch
import os
from pathlib import Path
from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer
from transformers import (
    RobertaTokenizerFast, RobertaConfig, RobertaForMaskedLM,
    DataCollatorForLanguageModeling, Trainer, TrainingArguments,
    LineByLineTextDataset
)
from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing

# 1. Setup logging
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

# 2. Load Wikipedia dataset
dataset_checkpoint = "wikimedia/wikipedia"
wiki = load_dataset(
    dataset_checkpoint,
    "20231101.vi",
    trust_remote_code=True,
    cache_dir=f'./cache/dataset/{dataset_checkpoint}'
)

# Save to plain text
output_dir = Path("./data/wikipedia_vi")
output_dir.mkdir(parents=True, exist_ok=True)
output_txt_path = output_dir / "wiki_corpus.txt"

with open(output_txt_path, "w", encoding="utf-8") as f:
    for example in wiki["train"]:
        text = example["text"].strip()
        if text:
            f.write(text.replace("\n", " ") + "\n")

# 3. Login Hugging Face and Wandb
huggingface_hub.login("hf_token")
wandb.login(key="wandb_token")