import os
import random
from argparse import ArgumentParser
from collections import Counter

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch import nn
from torch.utils.data import DataLoader

import lib
import models
from distill_dataset import DistillDataset

parser = ArgumentParser()


class DistillModule(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.model =  models.DistillEmb(n_chars=len(
            train_dataset.char2int), output_size=300, dropout=0.0)
        self.triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)

        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):

        x, pos_w2v, pos_ft, neg_w2v, neg_ft = batch
        z = self.model(x)

        wloss = self.triplet_loss(z, pos_w2v, neg_w2v)
        floss = self.triplet_loss(z, pos_ft, neg_ft)
        loss = (floss + wloss) / 2
        self.log("train_loss", loss)
        self.log("train_ft_loss", floss)
        self.log("train_w2v_loss", wloss)
        return loss

    def configure_optimizers(self):
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1, gamma=self.hparams.step_gamma)
        return [self.optimizer], [self.scheduler]

    def validation_step(self, batch, batch_idx):
        x, pos_w2v, pos_ft, neg_w2v, neg_ft = batch
        z = self.model(x)

        wloss = self.triplet_loss(z, pos_w2v, neg_w2v)
        floss = self.triplet_loss(z, pos_ft, neg_ft)
        loss = (floss + wloss) / 2
        self.log("val_loss", loss)
        self.log("val_ft_loss", floss)
        self.log("val_w2v_loss", wloss)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("DistillModule")
        parser.add_argument("--learning_rate", type=float, default=0.001)
        parser.add_argument("--step_gamma", type=float, default=0.95)
        return parent_parser

parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--neg_seq_len", type=int, default=32)
parser.add_argument("--train_ratio", type=float, default=0.9)

parser = DistillModule.add_model_specific_args(parser)
parser = pl.Trainer.add_argparse_args(parser)
args = parser.parse_args()
print(args.learning_rate)

logger = TensorBoardLogger("logs", name="distill")
early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=3, verbose=False, mode="min")
trainer = pl.Trainer.from_argparse_args(args, logger=logger)


batch_size = args.batch_size
neg_seq_length = args.neg_seq_len
train_ratio = args.train_ratio

fasttext_emb_path = "dataset/corpus/am-ft.vec"
word2vec_emb_path = "dataset/corpus/am-w2v.vec"
train_corpus_path = "dataset/corpus/clean-am-train-corpus.txt"
charset_path = "data/am-charset.txt"

ft_emb = lib.load_word_embeddings(fasttext_emb_path, word_prob=0.5) # load about 75% of the vectors
w2v_emb = lib.load_word_embeddings(word2vec_emb_path, target_words=ft_emb)

vocab = set(ft_emb.keys()).intersection(w2v_emb.keys())
if '</s>' in vocab:
    vocab.remove('</s>')
words = open(train_corpus_path, encoding='utf-8').read().split()
words = [word for word in words if word in vocab]

train_size = int(len(vocab) * train_ratio)
vocab = list(vocab)
np.random.shuffle(vocab)

train_vocab = vocab[:train_size]
test_vocab = vocab[train_size:]

print(f"Training vocab: {len(train_vocab)}, Test vocab: {len(test_vocab)}")
print(f"Training on {len(words)} words")
vocab2index = {v: k for k, v in enumerate(train_vocab)}
index2vocab = {k: v for k, v in enumerate(train_vocab)}

train_dataset = DistillDataset(words=words, vocab=train_vocab,
                               vocab2index=vocab2index,  w2v_vectors=w2v_emb, ft_vectors=ft_emb,
                               charset_path="data/am-charset.txt", neg_seq_len=neg_seq_length, max_word_len=13, pad_char='_')

test_dataset = DistillDataset(words=words,  vocab=test_vocab, vocab2index=vocab2index,
                              w2v_vectors=w2v_emb, ft_vectors=ft_emb,
                              charset_path="data/am-charset.txt", neg_seq_len=neg_seq_length, max_word_len=13, pad_char='_')


train_dataloader = DataLoader(
    train_dataset, shuffle=True,  batch_size=batch_size)
test_dataloader = DataLoader(
    test_dataset, batch_size=batch_size)

trainer.fit(model=DistillModule(**vars(args)),
            train_dataloaders=train_dataloader, val_dataloaders=test_dataloader)
