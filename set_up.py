import csv
import pandas
import os
import logging
import random
import numpy as np
import torch
from transformers import BertTokenizer

ADDITIONAL_SPECIAL_TOKENS = ["[CLS]", "[SEP]"]

def get_label(args):
    return [label.strip() for label in open(os.path.join(args.data_dir, args.label_file),'r',encoding='utf-8')]

def load_tokenizer(args):
    tokenizer = BertTokenizer.from_pretrained(args.model_name)
    tokenizer.add_special_tokens({"additional_special_tokens": ADDITIONAL_SPECIAL_TOKENS})
    return tokenizer

def init_logging():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )
def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)