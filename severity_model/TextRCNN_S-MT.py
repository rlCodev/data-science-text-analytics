# _*_ coding: utf-8 _*_

import argparse
import time

from nltk.tokenize import RegexpTokenizer
from collections import Counter

import gensim.downloader as api

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, f1_score


parser = argparse.ArgumentParser()
parser.add_argument('--dev_run', action='store_true')
parser.add_argument('--working_aspect_idx', type=int, default=0, help='Apect index as in [frightening, alcohol, nudity, violence, profanity].')
parser.add_argument('--base_dir', type=str, default='../data2021/splited_data/')
parser.add_argument('--model_save_dir', type=str, default='./TextRCNN_S-MT_save/')
parser.add_argument('--use_gpu_idx', type=int, default=0)

parser.add_argument('--train_batch_size', type=int, default=40, help='train_batch_size.')
parser.add_argument('--dev_batch_size', type=int, default=2, help='dev_batch_size.')
parser.add_argument('--test_batch_size', type=int, default=2, help='test_batch_size.')

parser.add_argument('--slate_num', type=int, default=2, help='compare num.')
parser.add_argument('--rank_output_size', type=int, default=3, help='rank output num.')
parser.add_argument('--cls_output_size', type=int, default=4, help='class num.')
parser.add_argument('--input_size', type=int, default=300, help='input dimension.')
parser.add_argument('--hidden_size', type=int, default=200, help='RNN hidden dimension.')
parser.add_argument('--projection_size', type=int, default=100, help='projection_size dimension.')

parser.add_argument('--lr', type=float, default=1e-3, help='learning rate.')
parser.add_argument('--training_epochs', type=int, default=200, help='Training epochs.')
parser.add_argument('--patience', type=int, default=10, help='Early stop patience.')
parser.add_argument('--multiple_runs', type=int, default=5, help='Multiple runs of experiment.')
parser.add_argument('--numpy_seed', type=int, default=42, help='NumPy seed.')

args = parser.parse_args()

doc_list = ['frightening', 'alcohol', 'nudity', 'violence', 'profanity']

working_aspect = doc_list[args.working_aspect_idx]

print('Now working on:', working_aspect)

train_file = args.base_dir + working_aspect + '_train.pkl'
dev_file = args.base_dir + working_aspect + '_dev.pkl'
test_file = args.base_dir + working_aspect + '_test.pkl'

train_data = pd.read_pickle(train_file)
dev_data = pd.read_pickle(dev_file)
test_data = pd.read_pickle(test_file)

to_device = 'cuda:' + str(args.use_gpu_idx)


### load pretrained word vectors ###

cur_time = time.time()
glove_model = api.load("glove-wiki-gigaword-300")
glove_dim = 300
print('load word emb takes time:', time.time() - cur_time)

np.random.seed(args.numpy_seed)
pad_emb = np.random.uniform(low=-0.1, high=0.1, size=(glove_dim,))
unk_emb = np.random.uniform(low=-0.3, high=0.3, size=(glove_dim,))

tokenizer = RegexpTokenizer('\w+|\$[\d\.]+|\S+')


def dfs_to_all_string_text(df_train, df_dev, df_test):
    big_df = pd.concat([df_train, df_dev, df_test])
    big_df_text_list = list(big_df['text'])
    list_of_text_string = [' '.join(each_text)
                           for each_text in big_df_text_list]
    all_text = ' '.join(list_of_text_string).lower()
    tk_all_text_list = tokenizer.tokenize(all_text)
    return tk_all_text_list


def create_dictionary(flat_text):
    sort_by_frequency = Counter(flat_text).most_common()
    unique_words = ['<PAD>', '<UNK>'] + [key for key, _ in sort_by_frequency]
    unique_words_idx = list(range(len(unique_words)))
    word_dic = dict(zip(unique_words, unique_words_idx))
    return word_dic, unique_words


def emb_look_up(word):
    try:
        emb = glove_model[word]
    except KeyError:
        emb = unk_emb
    if word == '<PAD>':
        emb = pad_emb
    return emb


def create_emb_dic(unique_words):
    word_emb_dic = {}
    idx_emb_dic = {}
    for i in range(len(unique_words)):
        # make 2 dic, one is word-emb, one is idx-embn
        emb_value = emb_look_up(unique_words[i])
        word_emb_dic[unique_words[i]] = emb_value
        idx_emb_dic[i] = emb_value
    return word_emb_dic, idx_emb_dic


def look_up(word_dic, word):
    value = word_dic.get(word, None)
    if value is None:
        value = 1  # unk
    return value


def process_text_to_input(one_text):
    # for further extension, use for loop instead of list comprehension
    text_idx = []
    # other word level feature list = []
    for each_word in one_text:
        # use lower case
        text_idx.append(look_up(word_dic, each_word.lower()))

    return text_idx


big_string = dfs_to_all_string_text(train_data, dev_data, test_data)
word_dic, unique_words = create_dictionary(big_string)
word_emb_dic, idx_emb_dic = create_emb_dic(unique_words)


# optional - create emb weights matrix

# def construct_emb_weights(idx_emb_dic):
#     emb_list = [idx_emb_dic[i] for i in range(len(idx_emb_dic))]
#     emb_weights_mat = np.stack(emb_list, axis=0)
#     return emb_weights_mat

# emb_weights_matrix = construct_emb_weights(idx_emb_dic)


def vec_look_up(word):
    try:
        emb = word_emb_dic[word]
    except KeyError:
        emb = unk_emb
    if word == '<PAD>':
        emb = pad_emb
    return emb


def get_column(matrix, i):
    return [row[i] for row in matrix]


def compare_pair(lst):
    result = 0
    # compare left to right, if left < right return 0...
    if lst[0] < lst[1]:
        result = 0
    elif lst[0] == lst[1]:
        result = 1
    else:
        result = 2
    return result


def label_to_pair_compare(lst):
    return torch.Tensor([compare_pair(each) for each in lst]).long()


class LSTM_model(pl.LightningModule):
    def __init__(self, input_size, slate_num, output_size, class_num, hidden_size, projection_size):
        super(LSTM_model, self).__init__()

        bsz = 1
        self.direction = 2
        self.input_size = input_size
        self.slate_num = slate_num
        self.output_size = output_size
        self.class_num = class_num
        self.hidden_size = hidden_size
        self.projection_size = projection_size
        self.batch_size = bsz

        self.lstm = nn.LSTM(
            self.input_size,
            self.hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.projection = nn.Linear(
            self.hidden_size * self.direction,
            self.projection_size
        )

        self.ranker = nn.Linear(
            self.projection_size * self.slate_num,
            self.output_size
        )

        self.classifier = nn.Linear(
            self.hidden_size * self.direction,
            self.class_num
        )

    def forward_one(self, x, batch_size=None):

        lens = [len(sq) for sq in x]
        x = pad_sequence(x, batch_first=True, padding_value=0)
        x = pack_padded_sequence(x, lens, batch_first=True, enforce_sorted=False)

        if batch_size is None:
            h_0 = torch.zeros(
                1 * self.direction,
                self.batch_size,
                self.hidden_size
            ).requires_grad_().to(device=to_device)

            c_0 = torch.zeros(
                1 * self.direction,
                self.batch_size,
                self.hidden_size).requires_grad_().to(device=to_device)

        else:
            h_0 = torch.zeros(
                1 * self.direction,
                batch_size,
                self.hidden_size
            ).requires_grad_().to(device=to_device)
            
            c_0 = torch.zeros(
                1 * self.direction,
                batch_size,
                self.hidden_size
            ).requires_grad_().to(device=to_device)

        output, (final_hidden_state, final_cell_state) = self.lstm(
            x, (h_0, c_0))
        output = pad_packed_sequence(output, batch_first=True)

        # max on seq length dimension, which is 1. (0 is batch, 2 is embedding)
        # after max, (max tensor, max_indices)
        output = torch.max(output[0], dim=1)[0]

        output_rank = self.projection(output)
        output_cls = self.classifier(output)

        return output_rank, output_cls

    def forward(self, x, batch_size=None):
        current_bsz = len(x)
        list_for_rank = []
        list_for_class = []
        # x shape: batch * ranklists
        # [[1,2],
        # [3,4],
        # [5,6],
        # ...
        # [99,100]]

        # print(len(x), len(x[0]))
        for i in range(len(x[0])):  # 2 dim -> # [[1,3,5],[2,4,6]]
            # one column is one batch, feed in one batch of
            rank_output, cls_output = self.forward_one(
                get_column(x, i), batch_size=current_bsz)
            list_for_rank.append(rank_output)  # one rank output
            # one column classification output = batch size * num_class
            list_for_class.append(cls_output)
        # 2 dim -> # [[1,3,5],[2,4,6]]
        championship = torch.cat(list_for_rank, dim=1)  # [[1,2],[3,4],[5,6]]

        final_rank_out = self.ranker(championship)
        final_cls_out = torch.stack(list_for_class)  # [[1,3,5],[2,4,6]]
        final_cls_out = final_cls_out.permute(1, 0, 2)
        # [num_samples * num_class]
        final_cls_out = final_cls_out.reshape(-1, self.class_num)

        return final_rank_out, final_cls_out

    def rank_loss_function(self, y_pred, y_true):

        return F.cross_entropy(y_pred, y_true)

    def cls_loss_function(self, prediction, target):
        return F.cross_entropy(prediction, target)

    def loss_function(self, rank_loss, cls_loss):
        return rank_loss + cls_loss

    def training_step(self, batch, batch_idx):
        text, target = batch
        # for classification, num_sample *1
        flat_target = torch.flatten(target).long()
        pairwise_target = label_to_pair_compare(target).to(target.device)

        rank_out, cls_out = self(text)

        rank_loss = self.rank_loss_function(rank_out, pairwise_target)
        cls_loss = self.cls_loss_function(cls_out, flat_target)

        loss = self.loss_function(rank_loss, cls_loss)

        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        text, target = batch
        # for classification, num_sample *1
        flat_target = torch.flatten(target).long()
        pairwise_target = label_to_pair_compare(target).to(target.device)
        rank_out, cls_out = self(text)
        # dummy batch

        rank_loss = self.rank_loss_function(rank_out, pairwise_target)
        cls_loss = self.cls_loss_function(cls_out, flat_target)

        prediction_digits = [torch.argmax(x).item() for x in cls_out]

        val_loss = self.loss_function(rank_loss, cls_loss)

        return {'prediction_digits': prediction_digits, 'target': flat_target.tolist(), 'val_loss': val_loss}

    def test_step(self, batch, batch_idx):
        text, target = batch
        # for classification, num_sample *1
        flat_target = torch.flatten(target).long()
        pairwise_target = label_to_pair_compare(target).to(target.device)
        rank_out, cls_out = self(text)

        rank_loss = self.rank_loss_function(rank_out, pairwise_target)
        cls_loss = self.cls_loss_function(cls_out, flat_target)

        prediction_digits = [torch.argmax(x).item() for x in cls_out]

        test_loss = self.loss_function(rank_loss, cls_loss)

        return {'prediction_digits': prediction_digits, 'target': flat_target.tolist(), 'test_loss': test_loss}

    def validation_epoch_end(self, val_step_outputs):
        avg_val_loss = torch.tensor([x['val_loss'] for x in val_step_outputs]).mean()
        val_predictions = [x['prediction_digits'] for x in val_step_outputs]
        val_targets = [x['target'] for x in val_step_outputs]
        # unpack list of lists
        val_predictions = [
            item for sublist in val_predictions for item in sublist]
        val_targets = [item for sublist in val_targets for item in sublist]

        cls_report = classification_report(val_targets, val_predictions, digits=4)
        print(cls_report)

        val_f1 = f1_score(val_targets, val_predictions, average='macro')

        return {'avg_val_loss': avg_val_loss, 'val_f1': val_f1}

    def test_epoch_end(self, test_step_outputs):
        avg_test_loss = torch.tensor([x['test_loss'] for x in test_step_outputs]).mean()
        test_predictions = [x['prediction_digits'] for x in test_step_outputs]
        test_targets = [x['target'] for x in test_step_outputs]
        # unpack list of lists
        test_predictions = [item for sublist in test_predictions for item in sublist]
        test_targets = [item for sublist in test_targets for item in sublist]

        cls_report = classification_report(test_targets, test_predictions, digits=4)
        print(cls_report)

        test_f1 = f1_score(test_targets, test_predictions, average='macro')

        return {'avg_test_loss': avg_test_loss, 'test_f1': test_f1}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=args.lr)

    def train_dataloader(self):
        train_raw_data = pd.read_pickle(train_file)
        train_dataset = MovieScriptDataset(train_raw_data)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.train_batch_size,
            shuffle=True,
            collate_fn=my_collate_fn,
            num_workers=10,
            drop_last=True)

        return train_loader

    def val_dataloader(self):
        dev_raw_data = pd.read_pickle(dev_file)
        dev_dataset = MovieScriptDataset(dev_raw_data)
        dev_loader = torch.utils.data.DataLoader(
            dev_dataset,
            batch_size=args.dev_batch_size,
            shuffle=False,
            collate_fn=my_collate_fn,
            drop_last=True)

        return dev_loader

    def test_dataloader(self):
        test_raw_data = pd.read_pickle(test_file)
        test_dataset = MovieScriptDataset(test_raw_data)
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.test_batch_size,
            shuffle=False,
            collate_fn=my_collate_fn,
            drop_last=True)

        return test_loader


class MovieScriptDataset(torch.utils.data.Dataset):
    def __init__(self, tabular):
        if isinstance(tabular, str):
            self.annotations = pd.read_csv(tabular, sep='\t')
        else:
            self.annotations = tabular

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        # -1 is text index, list of sent
        text = self.annotations.iloc[index, -1]
        text = ' '.join(text).lower()  # concate all sentences
        text = tokenizer.tokenize(text)  # tknz
        text = torch.tensor(
            np.stack([vec_look_up(each_word) for each_word in text]), dtype=torch.float32)

        y_label = torch.tensor(
            int(self.annotations.iloc[index, -2]))  # -2 is label index

        return {
            'text': text,
            'label': y_label
        }


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def my_collate_fn(batch):
    # batch size * 2 samples
    text_batch = [each_item['text'] for each_item in batch]

    # reshape to [batch size * 2]
    text_batch = list(chunks(text_batch, args.slate_num))

    label_batch = torch.stack([each_item['label']
                               for each_item in batch]).float()
    # reshape to [batch size * 2]
    label_batch = label_batch.reshape(-1, args.slate_num)

    return text_batch, label_batch


if __name__ == "__main__":

    for counting in range(args.multiple_runs):
        
        model = LSTM_model(
            args.input_size,
            args.slate_num,
            args.rank_output_size,
            args.cls_output_size,
            args.hidden_size,
            args.projection_size
        )

        early_stop_callback = EarlyStopping(
            monitor='val_f1',
            min_delta=0.00,
            patience=args.patience,
            verbose=False,
            mode='max'
        )

        checkpoint_callback = ModelCheckpoint(
            monitor='val_f1',
            filepath=args.model_save_dir + working_aspect,
            mode='max'
        )

        trainer = pl.Trainer(
            fast_dev_run=args.dev_run,
            max_epochs=args.training_epochs,
            gpus=[args.use_gpu_idx],
            callbacks=[early_stop_callback],
            checkpoint_callback=checkpoint_callback
        )

        trainer.fit(model)

        result = trainer.test()
        print(result)
