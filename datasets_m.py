import torch
from torch.utils.data import Dataset, DataLoader
import os
from set_up import *
import pytorch_lightning as pl

import logging
from pytorch_lightning.callbacks import ModelCheckpoint
logger = logging.getLogger(__name__)
class QTagDataset(Dataset):

    def __init__(self, text, tags, tokenizer, max_len):
        super().__init__()
        self.tokenizer = tokenizer
        self.text = text
        self.labels = tags
        self.max_len = max_len


    def __len__(self):
        return len(self.text)

    def __getitem__(self, item_idx):
        text = self.text[item_idx]
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,  # Add [CLS] [SEP]
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=False,
            return_attention_mask=True,  # Differentiates padded vs normal token
            truncation=True,  # Truncate data beyond max length
            return_tensors='pt'  # PyTorch Tensor format
        )

        input_ids = inputs['input_ids'].flatten()
        attn_mask = inputs['attention_mask'].flatten()
        # token_type_ids = inputs["token_type_ids"]

        return {
            'input_ids': input_ids,
            'attention_mask': attn_mask,
            'label': torch.tensor(self.labels[item_idx], dtype=torch.float)

        }


class SentiventDataModule(pl.LightningDataModule):

    def __init__(self, args, x_tr, y_tr, x_val, y_val, x_test, y_test, tokenizer):
        super().__init__()
        self.args = args
        self.tr_text = x_tr
        self.tr_label = y_tr
        self.val_text = x_val
        self.val_label = y_val
        self.test_text = x_test
        self.test_label = y_test
        self.tokenizer =tokenizer
        self.train_batch_size = self.args.train_batch_size
        self.test_batch_size = self.args.eval_batch_size
        self.max_token_len = self.args.max_seq_len

    def setup(self, stage=None):
        self.train_dataset = QQDataset(text=self.tr_text, tags=self.tr_label, tokenizer=self.tokenizer,
                                         max_len=self.max_token_len)
        self.val_dataset = QQDataset(text=self.val_text, tags=self.val_label, tokenizer=self.tokenizer,
                                       max_len=self.max_token_len)
        self.test_dataset = QQDataset(text=self.test_text, tags=self.test_label, tokenizer=self.tokenizer,
                                        max_len=self.max_token_len)
        return self.train_dataset, self.val_dataset, self.test_dataset

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.train_batch_size, shuffle=True, num_workers=16)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.test_batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.test_batch_size)

processors = {"sentivent" :  SentiventDataModule}
def tokenizing(args, yt, sentence):
    #processor = processors[args.task](args)  # semeval/wiki80/sentivent
    cached_file = os.path.join(args.data_dir, "cached_{}_{}_{}".format(args.task, list(
        filter(None, args.model_name.split("/"))).pop(), args.input_size))
    '''
    if os.path.exists(cached_file):
        logger.info("Loading features from cached file %s", cached_file)
        tok_sentences = torch.load(cached_file)
    else:
        logger.info("Loading features from dataset files at %s", args.data_dir)
    '''
    from sklearn.model_selection import train_test_split
    # First Split for Train and Test
    x_train, x_test, y_train, y_test = train_test_split(sentence, yt, test_size=0.1, random_state=args.seed,
                                                        shuffle=True)
    # Next split Train in to training and validation
    x_tr, x_val, y_tr, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=args.seed,
                                                shuffle=True)

    print(len(x_tr), len(x_val), len(x_test))


    # Initialize the Bert tokenizer
    Bert_tokenizer = load_tokenizer(args)

    max_word_cnt = 50
    sen_cnt = 0

    # For every sentence...
    for sen in sentence:

        # Tokenize the text and add `[CLS]` and `[SEP]` tokens.
        input_ids = Bert_tokenizer.encode(sen, add_special_tokens=True)

        # Update the maximum sentence length.
        if len(input_ids) > max_word_cnt:
            sen_cnt += 1

    print(f'# Question having word count > {max_word_cnt}: is  {sen_cnt}')



    return  x_tr, y_tr, x_val, y_val, x_test, y_test, Bert_tokenizer, cached_file

