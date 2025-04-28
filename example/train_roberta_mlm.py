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
