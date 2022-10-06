from argparse import ArgumentParser
import os
import warnings
from os import mkdir
import distill_emb_model

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.optim as optim
from seqeval.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

import crf_model
import lib
from predict import WordsTagger
from preprocess import *
from utils import *

warnings.filterwarnings("ignore")


class SeqPredModule(pl.LightningModule):
    def __init__(self, wordTagger: WordsTagger, embeder=None, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.preprocessor = preprocessor
        self.wordTagger = wordTagger
        self.model = crf_model.BiLSTMCRF(
            vocab_size=self.hparams.vocab_size,
            embeder=embeder,
            tagset_size=self.hparams.tagset_size,
            embedding_dim=self.hparams.embedding_dim,
            hidden_dim=self.hparams.hidden_dim,
            num_rnn_layers=self.hparams.num_rnn_layers,
            rnn=self.hparams.rnn_type,
            rnn_dropout=self.hparams.rnn_dropout,
            fc_dropout=self.hparams.fc_dropout,
            emb_dropout=self.hparams.emb_dropout,
            device=self.device)
        # self.mod

    def training_step(self, batch, batch_idx):

        xb, cs, yb = batch
        loss = self.model.loss(xb, cs, yb)
        # print(loss.device)
        f1_score = self.evaluate(batch, batch_idx)
        self.log("train_loss", loss)
        self.log("train_f1_score", f1_score)

        return loss

    def configure_optimizers(self):
        self.optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, 1, gamma=self.hparams.step_gamma)
        return [self.optimizer], [self.scheduler]

    def validation_step(self, batch, batch_idx):

        xb, cs, yb = batch
        loss = self.model.loss(xb, cs, yb)
        f1_score = self.evaluate(batch, batch_idx)
        self.log("val_loss", loss)
        self.log("val_f1_score", f1_score)
        return loss

    
    def evaluate(self, batch, batch_idx):

    
        word_ids, cs, labels = batch

        preds = self.wordTagger((word_ids, cs), self.model, device=self.device)
        xlabels = []
        for ir, row in enumerate(labels):
            s = [self.preprocessor.tags[x] for x in row]
            xlabels.append(s[:len(preds[0][ir])])


        # acc = accuracy_score(preds[0], xlabels)
        f1 = f1_score(preds[0], xlabels)
        # rec = recall_score(preds[0], xlabels)
        # pre = precision_score(preds[0], xlabels)

        # all_eval = [acc, pre, rec, f1]

        # by_lang = {k:[] for k in range(10)}

        # tby_lang = {k:[] for k in range(10)}
        # pby_lang = {k:[] for k in range(10)}

        # for i in range(len(all_langs)):
        #     idx = np.argmax(all_langs[i]).item()

        #     tby_lang[idx].append(all_targets[i])
        #     pby_lang[idx].append(xlabels)

        # for i in range(10):
        #     lang_targets = tby_lang[i]
        #     lang_pres = pby_lang[i]

        #     acc = accuracy_score(lang_pres, lang_targets)
        #     f1 = f1_score(lang_pres, lang_targets)
        #     rec = recall_score(lang_pres, lang_targets)
        #     pre = precision_score(lang_pres, lang_targets)

        #     by_lang[i] = [acc, pre, rec, f1]

        
        return f1 #, by_lang

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("SeqPred")
        parser.add_argument("--learning_rate", type=float, default=0.001)
        parser.add_argument('--embedding_dim', type=int, default=100,
                            help='the dimension of the embedding layer')
        parser.add_argument('--hidden_dim', type=int, default=128,
                            help='the dimension of the RNN hidden state')
        parser.add_argument('--num_rnn_layers', type=int,
                            default=2, help='the number of RNN layers')
        parser.add_argument('--rnn_type', type=str, default="lstm",
                            help='RNN type, choice: "lstm", "gru"')
        parser.add_argument('--weight_decay', type=float,
                            default=0., help='the L2 normalization parameter')
        parser.add_argument('--fc_dropout', type=float,
                            default=0.1, help='dropout on fc layer')
        parser.add_argument('--step_gamma', type=float,
                            default=0.9, help='stem gamma for lr schedular')
        parser.add_argument('--rnn_dropout', type=float,
                            default=0.1, help='dropout between rnn layer')
        parser.add_argument('--emb_dropout', type=float,
                            default=0.1, help='dropout on embedding layer')

        return parent_parser


parser = ArgumentParser()
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument('--vector_file', type=str, help="word embedding file")
parser.add_argument('--max_seq_len', type=int, default=200,
                    help='max sequence length within training')
parser.add_argument('--data_folder', type=str, default="ner",
                    help='RNN type, choice: "lstm", "gru"')
parser.add_argument('--charset_path', type=str, default="ner",
                    help='character set file')
parser.add_argument('--emb_type', type=str, default="CNN",
                    help='CNN or word_emb')
parser.add_argument('--emb_file', type=str, default="CNN",
                    help='path to fasttext or word2vec file')

parser = SeqPredModule.add_model_specific_args(parser)
parser = pl.Trainer.add_argparse_args(parser)
args = parser.parse_args()

char2int, int2char = lib.build_charset(args.charset_path, 0)

preprocessor = Preprocessor(config_dir=args.data_folder,
                            save_config_dir=args.data_folder)

(x_train, c_train, y_train), (x_val, c_val,  y_val), (x_test, c_test, y_test) = preprocessor.load_dataset(
    args.data_folder, max_seq_len=args.max_seq_len, max_word_len=13, pad_char='_', charset_path='data/am-charset.txt')
train_dataset = TensorDataset(x_train, c_train,  y_train)
sample_idxs = np.random.randint(0, len(train_dataset), 300)
test_train_dataset = torch.utils.data.Subset(train_dataset, sample_idxs)
test_trian_dl = DataLoader(test_train_dataset, batch_size=args.batch_size)
train_dl = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
valid_dl = DataLoader(TensorDataset(x_val, c_val, y_val),
                      batch_size=args.batch_size * 2)
test_dl = DataLoader(TensorDataset(x_test, c_test, y_test),
                     batch_size=args.batch_size * 2)

args.vocab_size = len(preprocessor.vocab)
args.tagset_size = len(preprocessor.tags)

print(f"Learning {args.vocab_size} vocabs and {args.tagset_size} tags")

wordTagger = WordsTagger(proc=preprocessor)


if args.emb_type == 'CNN':
    embeder = distill_emb_model.DistillEmb(n_chars=len(
        char2int), output_size=300, dropout=0.0)
    print("Using DistillEmb Embedding")
else:
    embeder = None
    if args.emb_file != None:
        if os.path.exists(args.emb_file):
            print("Loading word vectors: ", args.emb_file)
        else:
            print("Word vector doesn't exist: ", args.emb_file)

model = SeqPredModule(wordTagger=wordTagger,  embeder=embeder, **vars(args))

logger = TensorBoardLogger("logs", name="ner")
early_stop_callback = EarlyStopping(
    monitor="val_loss", min_delta=0.00, patience=3, verbose=False, mode="min")
trainer = pl.Trainer.from_argparse_args(
    args, logger=logger, callbacks=[early_stop_callback])


trainer.fit(model=model,
            train_dataloaders=train_dl, val_dataloaders=valid_dl)
