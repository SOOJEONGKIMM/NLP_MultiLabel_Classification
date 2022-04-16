# Huggingface transformers
import transformers
from transformers import BertModel,BertTokenizer,AdamW, get_linear_schedule_with_warmup
import pytorch_lightning as pl
import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel

import logging
from pytorch_lightning.callbacks import ModelCheckpoint
class MultiClassifier(pl.LightningModule):
    # Set up the classifier
    def __init__(self, args, n_classes=10, steps_per_epoch=None):
        super().__init__()
        self.args = args
        self.bert = BertModel.from_pretrained(args.model_name, return_dict=True)
        self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)  # outputs = number of labels
        self.steps_per_epoch = steps_per_epoch
        self.n_epochs = self.args.epochs
        self.lr = self.args.learning_rate
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self,b_input_ids, b_input_mask, b_labels):
        output = self.bert(input_ids=b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
        output = self.classifier(output.pooler_output)

        return output

    def training_step(self, batch, batch_idx):
        #input_ids = batch['input_ids']
        #attention_mask = batch['attention_mask']
        #labels = batch['label']
        input_ids, attention_mask, labels = batch
        outputs = self(input_ids, attention_mask)
        loss = self.criterion(outputs, labels)
        self.log('train_loss', loss, prog_bar=True, logger=True)

        return {"loss": loss, "predictions": outputs, "labels": labels}

    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['label']

        outputs = self(input_ids, attention_mask)
        loss = self.criterion(outputs, labels)
        self.log('val_loss', loss, prog_bar=True, logger=True)

        return loss

    def test_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['label']

        outputs = self(input_ids, attention_mask)
        loss = self.criterion(outputs, labels)
        self.log('test_loss', loss, prog_bar=True, logger=True)

        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr)
        warmup_steps = self.steps_per_epoch // 3
        total_steps = self.steps_per_epoch * self.n_epochs - warmup_steps

        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

        return [optimizer], [scheduler]