import csv
import pandas
import os
import logging
import random
import numpy as np
import pandas as pd
import torch
import sys
import re
from par_parser import parameter_parser
from set_up import *
sys.path.append(".")

from datasets_m import  SentiventDataModule, tokenizing
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from model import *
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import multiprocessing
from torch.utils.data import TensorDataset, DataLoader
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



def get_type_rawdata(args):
    return open(os.path.join(args.data_dir, "dataset_event_type.tsv"),'r',encoding='utf-8')
def get_subtype_rawdata(args):
    return open(os.path.join(args.data_dir, "dataset_event_subtype.tsv"),'r',encoding='utf-8')

if __name__ =='__main__':

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
            self.tokenizer = tokenizer
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
            return DataLoader(self.train_dataset, batch_size=self.train_batch_size, shuffle=True, num_workers=0)

        def val_dataloader(self):
            return DataLoader(self.val_dataset, batch_size=self.test_batch_size)

        def test_dataloader(self):
            return DataLoader(self.test_dataset, batch_size=self.test_batch_size)

    class QQDataset(Dataset):

        def __init__(self, text, tags, tokenizer, max_len):
            super(QQDataset,self).__init__()
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
                return_token_type_ids=True,
                return_attention_mask=True,  # Differentiates padded vs normal token
                truncation=True,  # Truncate data beyond max length
                return_tensors='pt'  # PyTorch Tensor format
            )

            input_ids = inputs['input_ids'].flatten()
            attn_mask = inputs['attention_mask'].flatten()
            token_type_ids = inputs["token_type_ids"]

            input_ids = input_ids.clone().detach()
            attn_mask = attn_mask.clone().detach()
            label = torch.tensor(self.labels[item_idx], dtype=torch.float).to(torch.long)
            return input_ids, attn_mask, label
            '''
                'input_ids': input_ids,
                'attention_mask': attn_mask,
                'label': torch.tensor(self.labels[item_idx], dtype=torch.float)

            }
            '''
    logger = logging.getLogger(__name__)
    #apply_async_callback()
    args = parameter_parser()
    init_logging()
    type_data = open('D:/2021-2_NLP_Relation\RL_Relation\sentivent_event_sentence_classification\data\processed\dataset_event_type.tsv','r',encoding='utf-8')
    #subtype_data = get_subtype_rawdata(args)
    #type_data=pd.DataFrame(type_data)
    data = type_data.readlines()
    column=data[0]
    datas = data[1:]
    column = column.split('\t')
    column[-1]=column[-1][:-1] #remove end \n
    new_datas=[]
    for line in datas:
        line = line.split('\t')
        line[-1]=line[-1][:-1]
        new_datas.append(line)
    type_data = {string : ['tmp'] for string in column}
    col = type_data.keys()
    for i in range(len(new_datas)):
        for idx,c in enumerate(col):
            type_data[c].append(new_datas[i][idx])
    type_data = pd.DataFrame(type_data)
    type_data = type_data.drop(type_data.index[0]) #remove tmp

    type_data.replace('[]',np.nan, inplace=True) #remove NaN
    type_data.dropna(subset=['types_event'], inplace=True, axis=0)

    id = [i for i in range(len(type_data))]
    type_data['Id'] = id

    tags = type_data['types_event'].value_counts().keys()
    # First group tags Id wise
    tags_unq = type_data['types_event'].unique()

    df_tags = type_data['types_event'].groupby(type_data['Id']).apply(lambda x: x.values).reset_index(name='tags')
    lst_tags = list(tags)
    df = type_data[['text','types_event']]
    df_tag_lst = list(type_data['types_event'])
    print(df.shape)

    x, y = [],[]
    for i in range(len(df['types_event'])):
        lower_txt = str(df['text'].iloc[i]).lower() #convert text to lower case
        x.append(lower_txt)
    y = list(df['types_event'])

    from sklearn.preprocessing import MultiLabelBinarizer

    mlb = MultiLabelBinarizer()
    yt = mlb.fit_transform(y)
    # Getting a sense of how the tags data looks like
    print(yt[0])
    print(mlb.inverse_transform(yt[0].reshape(1, -1)))
    print(mlb.classes_)

    # compute no. of words in each question
    sentence = x
    word_cnt = [len(s.split()) for s in sentence]
    # Plot the distribution
    plt.figure(figsize=[8, 5])
    plt.hist(word_cnt, bins=40)
    plt.xlabel('Word Count/Sentence')
    plt.ylabel('# of Occurences')
    plt.title("Frequency of Word Counts/Sentence")
    plt.show()

    #Generally looks like we should be good with a max word count = 50 since that should cover the text of most questions.
    #Getting a sense of how many questions have more that 50 words.
    ####new start###
    df['tags'] = df['types_event'].apply(lambda x: x[1:-1])
    df.head()
    top_tags = df['tags'].value_counts().keys()[0:50]
    # top_tags = df['tags'].value_counts().keys()

    top_tags

    id = [i for i in range(len(df))]
    df['Id'] = id

    df_tags = df.groupby('Id').apply(lambda x: x['tags'].values).reset_index(name='tags')
    df_tags.head()

    from bs4 import BeautifulSoup


    def pre_process(text):

        text = BeautifulSoup(text).get_text()

        # fetch alphabetic characters
        text = re.sub("[^a-zA-Z]", " ", text)

        # convert text to lower case
        text = text.lower()

        # split text into tokens to remove whitespaces
        tokens = text.split()

        return " ".join(tokens)


    type_data['clean_text'] = type_data['text'].apply(pre_process)

    df = pd.merge(type_data, df_tags, how='inner', on='Id')

    df = df[['clean_text', 'tags']]
    df.head()

    x = []  # To store the filtered clean_body values
    y = []  # to store the corresponding tags
    # Convert to list data type
    lst_top_tags = list(top_tags)

    for i in range(len(df['tags'])):
        temp = []
        for tag in df['tags'][i]:
            if tag in lst_top_tags:
                dup_tag = tag.split(', ') #remove duplicated tag in multilabel
                for v in dup_tag:
                    if v not in temp:
                        temp.append(v)

        if (len(temp) > 0):
            x.append(df['clean_text'][i])
            y.append(temp)

    from sklearn.preprocessing import MultiLabelBinarizer

    mlb = MultiLabelBinarizer()

    yt = mlb.fit_transform(y)
    yt.shape

    # Getting a sense of how the tags data looks like
    print(yt[0])
    print(mlb.inverse_transform(yt[0].reshape(1, -1)))
    print(mlb.classes_)

    questions = x
    word_cnt = [len(quest.split()) for quest in questions]
    # Plot the distribution
    plt.figure(figsize=[8, 5])
    plt.hist(word_cnt, bins=40)
    plt.xlabel('Word Count/Sentence')
    plt.ylabel('# of Occurences')
    plt.title("Frequency of Word Counts/Sentence")
    plt.show()



    x_tr, y_tr, x_val, y_val, x_test, y_test, Bert_tokenizer, cached_file = tokenizing(args, yt=yt, sentence=questions)
    # Instantiate and set up the data_module
    Sendata_module = SentiventDataModule(args, x_tr, y_tr, x_val, y_val, x_test, y_test, Bert_tokenizer)
    Sendata_module.setup()

    logger.info("Saving tok_sentences to cached_file %s", cached_file)
    torch.save(Sendata_module, cached_file)
    logger.info("Loading features from cached file %s", cached_file)
    tok_sentences = torch.load(cached_file)
    steps_per_epoch = len(x_tr) // args.train_batch_size

    from pytorch_pretrained_bert import BertTokenizer, BertForTokenClassification
    #model = MultiClassifier(args, n_classes=10, steps_per_epoch=steps_per_epoch)
    model = BertForTokenClassification.from_pretrained("bert-base-uncased", num_labels=18)
    # Initialize Pytorch Lightning callback for Model checkpointing

    # saves a file like: input/QTag-epoch=02-val_loss=0.32.ckpt
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',  # monitored quantity
        filename='MultiClass-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,  # save the top 3 models
        mode='min',  # mode of the monitored quantity  for optimization
    )
    use_cuda = torch.cuda.is_available()

    device = torch.device("cuda:0" if use_cuda else "cpu")
    if use_cuda:
        trainer = pl.Trainer(max_epochs=args.epochs, gpus=1, callbacks=[checkpoint_callback], progress_bar_refresh_rate=30, num_sanity_val_steps=0)
    else:
        trainer = pl.Trainer(max_epochs=args.epochs, callbacks=[checkpoint_callback], num_sanity_val_steps=0,
                             progress_bar_refresh_rate=30)
    '''
    trainer.fit(model, Sendata_module)

    trainer.test(model, datamodule=Sendata_module)

    model_path = checkpoint_callback.best_model_path
    '''
    FULL_FINETUNING = True
    if FULL_FINETUNING:
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.0}

        ]
    else:
        param_optimizer = list(model.classifier.named_parameters())
        optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    train_dataset = QQDataset(text=x_tr, tags=y_tr, tokenizer=Bert_tokenizer,
                                     max_len=18) #fixed max_len 50 to 18
    val_dataset = QQDataset(text=x_val, tags=y_val, tokenizer=Bert_tokenizer,
                                   max_len=18)
    test_dataset = QQDataset(text=x_test, tags=y_test, tokenizer=Bert_tokenizer,
                                    max_len=args.max_seq_len)

    from tqdm import tqdm
    model.to(device)
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=0)
    val_dataloader = DataLoader(val_dataset, batch_size=args.eval_batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=args.eval_batch_size)


    def flat_accuracy(preds, labels, masks):
        mask_flat = masks.flatten()
        pred_flat = np.argmax(preds, axis=2).flatten()[mask_flat == 1]
        labels_flat = labels.flatten()[mask_flat == 1]
        return np.sum(pred_flat == labels_flat) / len(labels_flat)

    epochs=12
    max_grad_norm = 1.0
    criterion = nn.BCEWithLogitsLoss()
    for iter, _ in tqdm(enumerate(range(epochs))):
        model.train()
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0

        for step, batch in enumerate(train_dataloader):
            #b_input_ids = batch['input_ids']
            #b_input_mask = batch['attention_mask']
            #b_labels = batch['label']
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            loss = model(input_ids=b_input_ids,  attention_mask=b_input_mask, labels=b_labels)

            tr_loss += loss.mean().item()
            loss.mean().backward()
            nb_tr_examples += b_input_ids.size(0)
            nb_tr_steps += 1
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
            optimizer.step()
            model.zero_grad()
        print("Train loss: {}".format(tr_loss / nb_tr_steps))


    # Size of Test set
    print(f'Number of Questions = {len(x_test)}')

    #setup test dataset for BERT
    from torch.utils.data import TensorDataset

    # Tokenize all questions in x_test
    input_ids = []
    attention_masks = []

    for quest in x_test:
        encoded_quest = Bert_tokenizer.encode_plus(
            quest,
            None,
            add_special_tokens=True,
            max_length=18, #300
            padding='max_length',
            return_token_type_ids=False,
            return_attention_mask=True,
            truncation=True,
            return_tensors='pt'
        )

        # Add the input_ids from encoded question to the list.
        input_ids.append(encoded_quest['input_ids'])
        # Add its attention mask
        attention_masks.append(encoded_quest['attention_mask'])

    # Now convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(y_test, dtype=torch.long)

    # Create the DataLoader.
    pred_data = TensorDataset(input_ids, attention_masks, labels)
    pred_sampler = SequentialSampler(pred_data)
    pred_dataloader = DataLoader(pred_data, sampler=pred_sampler, batch_size=args.eval_batch_size)

    #Prediction on test set
    flat_pred_outs = 0
    flat_true_labels = 0
    # Put model in evaluation mode
    model = model.to(device)  # moving model to cuda
    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps = 0

    # Tracking variables
    pred_outs, true_labels = [], []
    # i=0

    # Predict

    for batch in pred_dataloader:
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)

        # Unpack the inputs from our dataloader
        b_input_ids, b_attn_mask, b_labels = batch

        with torch.no_grad():
            # Forward pass, calculate logit predictions
            tmp_eval_loss = model(input_ids=b_input_ids,  attention_mask=b_attn_mask, labels=b_labels)
            pred_out = model(b_input_ids, b_attn_mask)
            pred_out = torch.sigmoid(pred_out)
            # Move predicted output and labels to CPU
            pred_out = pred_out.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            masks = b_attn_mask.to('cpu').numpy()


            tmp_eval_accuracy = flat_accuracy(pred_out, label_ids, masks)
            eval_loss += tmp_eval_loss.mean().item()
            eval_accuracy += tmp_eval_accuracy
            nb_eval_steps += 1
            #label_ids = b_labels.to('cpu').numpy()
            #masks = b_attn_mask.to('cpu').numpy()
        eval_loss = eval_loss / nb_eval_steps
        pred_outs.append(pred_out)
        true_labels.append(label_ids)
        print("Val_Loss: {}".format(eval_loss))
        print("Accuracy: {}".format(eval_accuracy / nb_eval_steps))

    # Combine the results across all batches.
    flat_pred_outs = np.concatenate(pred_outs, axis=0)
    flat_pred_outs.shape
    flat_pred_outs = flat_pred_outs[:,0,:]
    # Combine the correct labels for each batch into a single list.
    flat_true_labels = np.concatenate(true_labels, axis=0)

    # define candidate threshold values
    a = flat_pred_outs[flat_true_labels == 1].mean()
    threshold = np.arange(a, a+0.05, 0.01)
    threshold = np.arange(0.4, 0.51, 0.01)
    threshold


    # convert probabilities into 0 or 1 based on a threshold value
    def classify(pred_prob, thresh):
        y_pred = []

        for tag_label_row in pred_prob:
            temp = []
            for tag_label in tag_label_row:
                if tag_label >= thresh:
                    temp.append(1)  # Infer tag value as 1 (present)
                else:
                    temp.append(0)  # Infer tag value as 0 (absent)
            y_pred.append(temp)

        return y_pred


    from sklearn import metrics

    scores = []  # Store the list of f1 scores for prediction on each threshold

    # convert labels to 1D array
    y_true = flat_true_labels.ravel()

    for thresh in threshold:
        # classes for each threshold
        pred_bin_label = classify(flat_pred_outs, thresh)

        # convert to 1D array
        y_pred = np.array(pred_bin_label).ravel()

        scores.append(metrics.f1_score(y_true, y_pred))

    # find the optimal threshold
    opt_thresh = threshold[scores.index(max(scores))]
    print(f'Optimal Threshold Value = {opt_thresh}')

    y_pred_labels = classify(flat_pred_outs, opt_thresh)

    #MR = np.all(int(pred_bin_label== np.array(flat_true_labels))).mean()
    #y_pred_labels = np.where(flat_pred_outs[0]==flat_pred_outs[0][flat_true_labels[0]==1])
    y_pred = np.array(y_pred_labels).ravel()  # Flatten
    target_names = ['class 0', 'class 1']
    print(metrics.classification_report(y_true, y_pred,  target_names=target_names))

    y_pred = mlb.inverse_transform(np.array(y_pred_labels))
    y_act = mlb.inverse_transform(flat_true_labels)
    matching=[]
    for i in range(len(new_datas)):
        if new_datas[i][1] in new_datas[i][-1]:
            matching.append(1)
        else:
            matching.append(0)
    for p in y_pred:
        for t in y_act:
            #print(p, t)
            if t in p:
                matching.append(1)
    p_df = pd.DataFrame({'Body': x_test, 'Actual Tags': y_act, 'Predicted Tags': y_pred})

    #p_df.to_csv('multilabel_result.csv')

    '''
    def emr(y_true, y_pred):
        n = len(y_true)
        row_indicators = np.all(y_true == y_pred, axis=1)  # axis = 1 will check for equality along rows.
        exact_match_count = np.sum(row_indicators)
        return exact_match_count / n


    emr(flat_true_labels, np.array(y_pred_labels))
    '''