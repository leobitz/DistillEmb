from argparse import ArgumentParser
import os
import warnings
from os import mkdir
from distill_dataset import UNK_WORD
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

import seq_model
import lib
from predict import WordsTagger
from preprocess import *
from utils import *
import sys

warnings.filterwarnings("ignore")


class SeqPredModule(pl.LightningModule):
    def __init__(self, wordTagger: WordsTagger, word2index=None, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.preprocessor = preprocessor
        self.wordTagger = wordTagger
        if args.emb_type != 'CNN':
            self.model = seq_model.BiLSTMCRF(
                vocab_size=self.hparams.vocab_size,
                tagset_size=self.hparams.tagset_size,
                input_size=self.hparams.embedding_dim,
                hidden_size=self.hparams.hidden_dim,
                num_rnn_layers=self.hparams.num_rnn_layers,
                rnn=self.hparams.rnn_type,
                word2index=word2index,
                rnn_dropout=self.hparams.rnn_dropout,
                fc_dropout=self.hparams.fc_dropout,
                emb_dropout=self.hparams.emb_dropout)
        else:
            self.model = seq_model.DistillBiLSTMCRF(
                tagset_size=self.hparams.tagset_size,
                input_size=self.hparams.embedding_dim,
                hidden_size=self.hparams.hidden_dim,
                charset_path=self.hparams.charset_path,
                num_rnn_layers=self.hparams.num_rnn_layers,
                rnn=self.hparams.rnn_type,
                rnn_dropout=self.hparams.rnn_dropout,
                fc_dropout=self.hparams.fc_dropout,
                emb_dropout=self.hparams.emb_dropout)

    def training_step(self, batch, batch_idx):

        xb, cs, yb, mask = batch
        loss = self.model.loss(xb, cs, yb)
        acc, f1, pre, rec  = self.evaluate(batch, batch_idx)
        self.log("train_loss", loss)
        self.log("train_f1", f1)
        self.log("train_acc", acc)
        self.log("train_pre", pre)
        self.log("train_rec", rec)

        return loss

    def configure_optimizers(self):
        self.optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, 1, gamma=self.hparams.step_gamma)
        return [self.optimizer], [self.scheduler]

    def validation_step(self, batch, batch_idx):

        xb, cs, yb, mask = batch
        loss = self.model.loss(xb, cs, yb)
        acc, f1, pre, rec = self.evaluate(batch, batch_idx)
        self.log("val_loss", loss)
        self.log("val_f1", f1)
        self.log("val_acc", acc)
        self.log("val_pre", pre)
        self.log("val_rec", rec)

        return loss

    
    def evaluate(self, batch, batch_idx):

    
        word_ids, cs, labels, mask_idx = batch

        preds = self.wordTagger((word_ids, cs), self.model, device=self.device)
        xlabels = []
        for ir, row in enumerate(labels):
            s = [self.preprocessor.tags[x] for x in row]
            xlabels.append(s[:len(preds[0][ir])])


        acc = accuracy_score(preds[0], xlabels)
        f1 = f1_score(preds[0], xlabels)
        rec = recall_score(preds[0], xlabels)
        pre = precision_score(preds[0], xlabels)

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

        
        return acc, f1, pre, rec #, by_lang

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("SeqPred")
        parser.add_argument("--learning-rate", type=float, default=0.001)
        parser.add_argument('--embedding-dim', type=int, default=100,
                            help='the dimension of the embedding layer')
        parser.add_argument('--hidden-dim', type=int, default=128,
                            help='the dimension of the RNN hidden state')
        parser.add_argument('--num-rnn-layers', type=int,
                            default=2, help='the number of RNN layers')
        parser.add_argument('--rnn-type', type=str, default="lstm",
                            help='RNN type, choice: "lstm", "gru"')
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
        parser.add_argument('--emb-type', type=str, default="CNN",
                    help='CNN or word_emb')
        return parent_parser


parser = ArgumentParser()
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument('--vector-file', type=str, help="word embedding file")
parser.add_argument('--max-seq-len', type=int, default=200,
                    help='max sequence length within training')

parser.add_argument('--charset-path', type=str, 
                    help='character set file')
parser.add_argument('--exp-name', type=str, 
                    help='experiment name')
parser.add_argument('--dataset-folder', type=str, 
                    help='datatset folder')


parser.add_argument('--trial-id', type=int, default=1,
                    help='model save folder name')
parser.add_argument('--train-model', action='store_true',
                    help='flag to test instead of train')
parser.add_argument('--no-train-model', action='store_false',
                    help='flag to test instead of train')
parser.add_argument('--test-trial-ids', type=str,
                    help='trial ids to test separated by -')

parser.add_argument("--grad-accumulate", type=int, default=1)
parser.add_argument('--data-size', type=float,
                            default=1.0, help='downstream data size in %')
parser.add_argument('--vocab-file', type=str,
                    help='File containing all the vocabs for the target task')


parser = SeqPredModule.add_model_specific_args(parser)
parser = pl.Trainer.add_argparse_args(parser)
args = parser.parse_args()

char2int, int2char = lib.build_charset(args.charset_path, 0)

preprocessor = Preprocessor(config_dir=args.dataset_folder,
                            save_config_dir=args.dataset_folder)

(x_train, c_train, y_train, m_train), (x_val, c_val,  y_val, m_val), (x_test, c_test, y_test, m_test) = preprocessor.load_dataset(
    args.dataset_folder, max_seq_len=args.max_seq_len, max_word_len=13, pad_char='_', charset_path=args.charset_path)
train_dataset = TensorDataset(x_train, c_train,  y_train, m_train)
sample_idxs = np.random.randint(0, len(train_dataset), 300)
test_train_dataset = torch.utils.data.Subset(train_dataset, sample_idxs)
test_trian_dl = DataLoader(test_train_dataset, batch_size=args.batch_size)
train_dl = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
valid_dl = DataLoader(TensorDataset(x_val, c_val, y_val, m_val),
                      batch_size=args.batch_size * 2)
test_dl = DataLoader(TensorDataset(x_test, c_test, y_test, m_test),
                     batch_size=args.batch_size * 2)

args.vocab_size = len(preprocessor.vocab)
args.tagset_size = len(preprocessor.tags)

print(f"Learning {args.vocab_size} vocabs and {args.tagset_size} tags")

wordTagger = WordsTagger(proc=preprocessor)


if args.emb_type != 'CNN':
    if args.vector_file == None or not os.path.exists(args.vector_file):
        sys.exit("Embedding file doesn't exist")
else:
    if args.vector_file == None:
        sys.exit('DistillEmb model doesnt exits')
        

args.test_models = not args.train_model

m = SeqPredModule(wordTagger=wordTagger, word2index=preprocessor.vocab_dict, **vars(args))

if args.emb_type != "CNN":
    word2vec = lib.load_word_embeddings(args.vector_file, target_words=set(preprocessor.vocab), header=False)
    n_loaded = m.model.init_emb(w2v=word2vec)
    print("Loaded embs in %", n_loaded * 100 / len(preprocessor.vocab))    
else:
    if args.vector_file != "scratch":
        checkpoint = torch.load(args.vector_file)
        model_state = checkpoint['state_dict']
        keys = list(model_state.keys())
        for key in keys:
            v = model_state.pop(key)
            model_state[key[6:]] = v
        m.model.embedding.load_state_dict(model_state)
        print("Loaded: ", args.vector_file)


print("Test mode:", args.test_models)
if not args.test_models:
    logger = TensorBoardLogger("logs", name=args.exp_name)
    
    trainer = pl.Trainer.from_argparse_args(args, logger=logger)

    trainer.fit(model=m,
            train_dataloaders=train_dl, val_dataloaders=valid_dl)
