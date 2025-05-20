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

# 4. Train tokenizer
data_path = './data/wikipedia_vi'
paths = [str(x) for x in Path(data_path).rglob("*.txt")]

tokenizer = ByteLevelBPETokenizer()
tokenizer.train(
    files=paths,
    vocab_size=152000,
    min_frequency=4,
    special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
)

# Save tokenizer
tokenizer_save_dir = Path("./roberta-mlm-tokenizer-v1")
tokenizer_save_dir.mkdir(parents=True, exist_ok=True)
tokenizer.save_model(str(tokenizer_save_dir))

# Load tokenizer with processors
tokenizer = ByteLevelBPETokenizer(
    "./roberta-mlm-tokenizer-v1/vocab.json",
    "./roberta-mlm-tokenizer-v1/merges.txt",
)
tokenizer._tokenizer.post_processor = BertProcessing(
    ("</s>", tokenizer.token_to_id("</s>")),
    ("<s>", tokenizer.token_to_id("<s>")),
)
tokenizer.enable_truncation(max_length=1024)

# Load into Hugging Face format
tokenizer = RobertaTokenizerFast.from_pretrained("./roberta-mlm-tokenizer-v1", max_len=512)

# 5. Create Roberta model
config = RobertaConfig(
    vocab_size=152_000,
    max_position_embeddings=514,
    num_attention_heads=12,
    num_hidden_layers=6,
    type_vocab_size=1,
)

model = RobertaForMaskedLM(config=config)
logger.info(f"Model parameters: {model.num_parameters()}")

# 6. Prepare dataset

dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path=str(output_txt_path),
    block_size=512,
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15
)

# 7. Setup Trainer
training_args = TrainingArguments(
    output_dir="./roberta-mlm-model-v1",
    overwrite_output_dir=True,
    evaluation_strategy="steps",
    eval_steps=5000,
    logging_strategy="steps",
    logging_steps=500,
    save_strategy="steps",
    save_steps=10000,
    save_total_limit=3,
    num_train_epochs=5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    gradient_accumulation_steps=1,
    learning_rate=5e-4,
    warmup_steps=1000,
    weight_decay=0.01,
    prediction_loss_only=True,
    fp16=True,
    push_to_hub=True,
    report_to="wandb",
    hub_model_id="Kudod/roberta-mlm-model-v1",
    hub_strategy="checkpoint",
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
    eval_dataset=dataset,
    tokenizer=tokenizer
)
