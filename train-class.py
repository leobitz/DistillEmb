
from argparse import ArgumentParser

import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from torch.utils.data import DataLoader

from class_model import  create_model
from distill_dataset import ClassificationDataset, UNK_WORD, collate_fun
import lib
from pytorch_lightning.callbacks import ModelCheckpoint
import os
import numpy as np

class ClassifyModule(pl.LightningModule):

    def __init__(self, class_indexices, word2index, **kwargs) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=['class_indexices', "word2index", "charset_path", "test_models", "test_trail_ids"])
        self.model = create_model(self.hparams, word2index)

        self.criterion = nn.CrossEntropyLoss()
        self.class_indexices = class_indexices

    def training_step(self, batch, batch_idx):

        x, y, seq_len = batch
        preds = self.model(x, seq_len)
        loss = self.criterion(preds, y)

        preds = torch.argmax(preds.detach().cpu(), dim=1).numpy()
        targets = y.detach().cpu().numpy()
        acc, pre, rec, f1 = self._report(preds, targets, self.class_indexices)

        self.log("train_loss", loss)
        self.log("train_f1", f1)
        self.log("train_precision", pre)
        self.log("train_recall", rec)
        self.log("train_acc", acc)

        return {"loss": loss, "f1": f1}

    def configure_optimizers(self):
        self.optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, 1, gamma=self.hparams.step_gamma)
        return [self.optimizer], [self.scheduler]

    def validation_step(self, batch, batch_idx):

        x, y, seq_len = batch
        preds = self.model(x, seq_len)
        loss = self.criterion(preds, y)

        preds = torch.argmax(preds.detach().cpu(), dim=1).numpy()
        targets = y.detach().cpu().numpy()
        acc, pre, rec, f1 = self._report(
            preds, targets, class_indexices=self.class_indexices)

        self.log("val_loss", loss)
        self.log("val_f1", f1)
        self.log("val_precision", pre)
        self.log("val_recall", rec)
        self.log("val_acc", acc)

        return {"loss": loss, "f1": f1}

    def test_step(self, batch, batch_idx):

        x, y, seq_len = batch
        preds = self.model(x, seq_len)
        loss = self.criterion(preds, y)

        preds = torch.argmax(preds.detach().cpu(), dim=1).numpy()
        targets = y.detach().cpu().numpy()
        acc, pre, rec, f1 = self._report(
            preds, targets, class_indexices=self.class_indexices)

        self.log("test_loss", loss)
        self.log("test_f1", f1)
        self.log("test_precision", pre)
        self.log("test_recall", rec)
        self.log("test_acc", acc)

        return {"loss": loss, "f1": f1}

    def _report(self, Y_preds, Y_test, class_indexices):
        pre = precision_score(
            Y_test, Y_preds, labels=class_indexices, average='macro', zero_division=0)
        rec = recall_score(Y_test, Y_preds, labels=class_indexices,
                           average='macro', zero_division=0)
        f1 = f1_score(Y_test, Y_preds, labels=class_indexices,
                      average='macro', zero_division=0)
        acc = accuracy_score(Y_test, Y_preds)
        return acc, pre, rec, f1

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("SeqPred")
        parser.add_argument("--learning-rate", type=float, default=0.001)
        parser.add_argument('--embedding-dim', type=int, default=100,
                            help='the dimension of the embedding layer')
        parser.add_argument('--hidden-dim', type=int, default=128,
                            help='the dimension of the RNN hidden state')
        parser.add_argument('--num-rnn-layers', type=int,
                            default=1, help='the number of RNN layers')
        parser.add_argument('--weight-decay', type=float,
                            default=0., help='the L2 normalization parameter')
        parser.add_argument('--fc-dropout', type=float,
                            default=0.1, help='dropout on fc layer')
        parser.add_argument('--step-gamma', type=float,
                            default=0.9, help='stem gamma for lr schedular')
        parser.add_argument('--rnn-dropout', type=float,
                            default=0.1, help='dropout between rnn layer')
        parser.add_argument('--emb-dropout', type=float,
                            default=0.1, help='dropout on embedding layer')
        parser.add_argument("--word2vec",  type=str,
                            help="word embedding file path")
        parser.add_argument('--emb-type', type=str, default="CNN",
                    help='CNN or word_emb')
        return parent_parser


parser = ArgumentParser()
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument('--vector-file', type=str, help="word embedding file")
parser.add_argument('--max-seq-len', type=int, default=200,
                    help='max sequence length within training')
parser.add_argument('--data-folder', type=str, default="ner",
                    help='RNN type, choice: "lstm", "gru"')
parser.add_argument('--charset-path', type=str, default="ner",
                    help='character set file')
parser.add_argument('--emb-file', type=str, default="CNN",
                    help='path to fasttext or word2vec file')
parser.add_argument('--exp-name', type=str, default="CNN",
                    help='experiment name')
parser.add_argument('--dataset-folder', type=str, default="tig",
                    help='datatset folder')
parser.add_argument('--trail-id', type=int, default=1,
                    help='model save folder name')
parser.add_argument('--train-model', action='store_true',
                    help='flag to test instead of train')
parser.add_argument('--no-train-model', action='store_false',
                    help='flag to test instead of train')
                    
parser.add_argument('--test-trail-ids', type=str,
                    help='trial ids to test separated by -')

parser = ClassifyModule.add_model_specific_args(parser)
parser = pl.Trainer.add_argparse_args(parser)
args = parser.parse_args()

args.test_models = not args.train_model

train_file = f"{args.dataset_folder}/clean-train.csv"
if args.test_models:
    test_file = f"{args.dataset_folder}/clean-dev.csv"
else:
    test_file = f"{args.dataset_folder}/clean-dev.csv"

train_data = pd.read_csv(train_file).to_numpy()
test_data = pd.read_csv(test_file).to_numpy()

class_labels = sorted(set(train_data[:, 0]))
print("class_labels", class_labels)


words = [UNK_WORD]
max_seq_len = 0
for line in train_data[:, 1]:
    line_words = line.split(' ')
    if len(line_words) > max_seq_len:
        max_seq_len = len(line_words)
    words.extend(line_words)

vocab = set(words)
word2index = {v: k for k, v in enumerate(vocab)}
label2index = {v: k for k, v in enumerate(class_labels)}

print("max_seq_len", max_seq_len)
if args.max_seq_len > max_seq_len:
    args.max_seq_len = max_seq_len
print("using max_seq_len", args.max_seq_len)

train_dataset = ClassificationDataset(data_rows=train_data, word2index=word2index, label2index=label2index,
                                      charset_path=args.charset_path, pad_char=' ', max_len=args.max_seq_len, word_output=(args.emb_type != "CNN"))

test_dataset = ClassificationDataset(data_rows=test_data, word2index=word2index, label2index=label2index,
                                     charset_path=args.charset_path, pad_char=' ',  max_len=args.max_seq_len, word_output=(args.emb_type != "CNN"))

print(train_data.shape, test_data.shape)

train_dataloader = DataLoader(
    train_dataset,   shuffle=True, collate_fn=collate_fun, num_workers=0, batch_size=args.batch_size, drop_last=True)
test_dataloader = DataLoader(
    test_dataset,  shuffle=False, collate_fn=collate_fun, num_workers=0, batch_size=args.batch_size, drop_last=False)

args.num_classes = len(class_labels)
args.train_embedding = True
args.vocab_size = len(vocab)

checkpoint_cb = ModelCheckpoint(
    save_top_k=-1,
    every_n_epochs=1,
    dirpath=f'saves/{args.exp_name}/{args.trail_id}',
    filename='{epoch}-{val_loss:.5f}-{val_f1:.5f}')
logger = TensorBoardLogger("logs", name=args.exp_name)

trainer = pl.Trainer.from_argparse_args(args)

m = ClassifyModule(list(label2index.values()), word2index, **vars(args))

if args.emb_type != "CNN" and args.vector_file != None:
    word2vec = lib.load_word_embeddings(args.vector_file, target_words=vocab)
    n_loaded = m.model.init_emb(w2v=word2vec)
    print("Loaded embs in %", n_loaded * 100 / len(vocab))

if not args.test_models:
    trainer.fit(model=m,
                train_dataloaders=train_dataloader,
                val_dataloaders=test_dataloader)
else:
    ids = args.test_trail_ids.split("-")
    trail_accs = []
    max_model_name = None
    max_val_acc = 0
    for idx in ids:
        folder_name = f'saves/{args.exp_name}/{idx}'
        names = sorted(os.listdir(folder_name), key=lambda x: int(x.split('-')[0].split('=')[1]))
        accs = [float(x[-12:-5]) for x in names]
        max_index = np.argmax(accs)
        if accs[max_index] > max_val_acc:
            max_val_acc = accs[max_index]
            max_model_name = f"{folder_name}/{names[max_index]}"
        trail_accs.append(accs)
    trail_accs = np.array(trail_accs)
    mean_trail_accs = trail_accs.mean(axis=0)
    max_epoch = mean_trail_accs.argmax()
    args.word2index = word2index
    args.class_indexices = list(label2index.values())
    test_accs = []
    for idx in ids:
        folder_name = f'saves/{args.exp_name}/{idx}'
        names = sorted(os.listdir(folder_name), key=lambda x: int(x.split('-')[0].split('=')[1]))
        name = names[max_epoch]
        path = f"{folder_name}/{name}"
        checkpoint = torch.load(path)
        print(path)
        # result = trainer.test(model=m, dataloaders=test_dataloader)
        # m.load_state_dict(checkpoint['state_dict'])
        # result = trainer.test(model=m, dataloaders=test_dataloader)
        m = ClassifyModule.load_from_checkpoint(path, **vars(args))
        result = trainer.validate(m, dataloaders=test_dataloader)
        print(result)