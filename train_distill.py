import random
from argparse import ArgumentParser

import os
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch import nn
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
import lib
from torch.optim.lr_scheduler import CosineAnnealingLR
import distill_emb_model
from distill_dataset import DistillDataset
from pytorch_lightning.loggers import WandbLogger
from create_model import create_distill_emb

random.seed(1000)
torch.random.manual_seed(10000)
np.random.seed(1000)
parser = ArgumentParser()


class DistillModule(pl.LightningModule):
    def __init__(self, char2int, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.model =  create_distill_emb(char2int=char2int, 
                                        dropout=0.0, 
                                        output_size=300, 
                                        pad_char=' ', 
                                        model_size=self.hparams.model_size)
        # self.save_hyperparameters(ignore=["char2int"])
        self.triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
        # print(self.model)
        

    def training_step(self, batch, batch_idx):

        x, pos_w2v, pos_ft, neg_w2v, neg_ft = batch
        z = self.model(x)

        wloss = self.triplet_loss(z, pos_w2v, neg_w2v)
        floss = self.triplet_loss(z, pos_ft, neg_ft)
        loss = (floss + wloss) / 2

        self.log("train_loss", loss)
        self.log("train_ft_loss", floss)
        self.log("train_w2v_loss", wloss)
        return {"loss": loss, "loss-w2v": wloss, "loss-ft": floss}

    def training_epoch_end(self, outputs) -> None:
        loss = sum(output['loss'] for output in outputs) / len(outputs)
        wloss = sum(output['loss-w2v'] for output in outputs) / len(outputs)
        floss = sum(output['loss-ft'] for output in outputs) / len(outputs)
        self.log("epoch_train_loss", loss)
        self.log("epoch_train_loss_ft", floss)
        self.log("epoch_train_loss_w2v", wloss)

    def validation_step(self, batch, batch_idx):
        x, pos_w2v, pos_ft, neg_w2v, neg_ft = batch
        z = self.model(x)

        wloss = self.triplet_loss(z, pos_w2v, neg_w2v)
        floss = self.triplet_loss(z, pos_ft, neg_ft)
        loss = (floss + wloss) / 2
        self.log("epoch_val_loss", loss)
        self.log("epoch_val_ft_loss", floss)
        self.log("epoch_val_w2v_loss", wloss)


    def configure_optimizers(self):
        
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        self.scheduler = CosineAnnealingLR(self.optimizer ,
                              T_max = self.hparams.total_iteration, # Maximum number of iterations.
                             eta_min = 1e-5) # Minimum learning rate.
        
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1, gamma=self.hparams.step_gamma)
        return [self.optimizer], [self.scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("DistillModule")
        parser.add_argument("--learning_rate", type=float, default=0.001)
        parser.add_argument("--step_gamma", type=float, default=0.90)
        parser.add_argument('--model-size', type=str,
                            default='small', help='model width: small (512) and large(768)')
        return parent_parser

def main():

    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--neg_seq_len", type=int, default=32)
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--vector-load-ratio", type=float, default=0.9)

    parser.add_argument('--fasttext-path', type=str, 
                        help='path to fasttext  file')
    parser.add_argument('--word2vec-path', type=str, 
                        help='path to fasttext  file')
    parser.add_argument('--charset-path', type=str, 
                        help='character set file')
    parser.add_argument('--exp-name', type=str, default="distill",
                        help='experiment name')
    parser.add_argument('--corpus', type=str, default="distill",
                        help='corpus file')

    parser.add_argument('--early-stop', type=int, default=0,  help='1 for early stop, 0 for max epoch train')

    parser = DistillModule.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    logger = TensorBoardLogger("logs", name=args.exp_name)
    cbs = []
    if args.early_stop == 1:
        early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=10, verbose=False, mode="min")
        cbs.append(early_stop_callback)

    checkpoint_cb = ModelCheckpoint(
        save_top_k=-1,
        every_n_epochs=4,
        dirpath=f'saves/{args.exp_name}',
        filename='{epoch}-{val_loss:.5f}-{val_f1:.5f}')
    cbs.append(checkpoint_cb)

    trainer = pl.Trainer.from_argparse_args(args, 
                            logger=logger, 
                            devices=1,
                            callbacks=cbs)


    batch_size = args.batch_size
    neg_seq_length = args.neg_seq_len
    train_ratio = args.train_ratio

    fasttext_emb_path = args.fasttext_path
    word2vec_emb_path = args.word2vec_path 
    train_corpus_path = args.corpus
    charset_path = args.charset_path 


    ft_emb = lib.load_word_embeddings(fasttext_emb_path, word_prob=args.vector_load_ratio) # load about 50% of the vectors
    print("FT loaded")
    w2v_emb = lib.load_word_embeddings(word2vec_emb_path, target_words=ft_emb)
    print("w2v loaded")
    vocab = set(ft_emb.keys()).intersection(w2v_emb.keys())
    if '</s>' in vocab:
        vocab.remove('</s>')
    print("Finished loading vectors")
    words = lib.load_corpus_words(train_corpus_path, line_prob=args.vector_load_ratio)
    words = [word for word in words if word in vocab]
    print("# of tokens: ", len(words))
    print("# of vocab: ", len(vocab))

    train_size = int(len(vocab) * train_ratio)
    vocab = list(vocab)
    np.random.shuffle(vocab)

    train_vocab = vocab[:train_size]
    test_vocab = vocab[train_size:]

    print(f"Training vocab: {len(train_vocab)}, Test vocab: {len(test_vocab)}")
    print(f"Training on {len(words)} words")
    vocab2index = {v: k for k, v in enumerate(train_vocab)}
    # index2vocab = {k: v for k, v in enumerate(train_vocab)}

    train_dataset = DistillDataset(words=words, vocab=train_vocab,
                                vocab2index=vocab2index,  w2v_vectors=w2v_emb, ft_vectors=ft_emb,
                                charset_path=args.charset_path, neg_seq_len=neg_seq_length, max_word_len=13, pad_char=' ')

    test_dataset = DistillDataset(words=words,  vocab=test_vocab, vocab2index=vocab2index,
                                w2v_vectors=w2v_emb, ft_vectors=ft_emb,
                                charset_path=args.charset_path, neg_seq_len=neg_seq_length, max_word_len=13, pad_char=' ')

    args.total_iteration = args.max_epochs * len(train_dataset) // batch_size
    train_dataloader = DataLoader(
        train_dataset, 
        num_workers=0, 
        pin_memory=True,
        shuffle=True,  
        batch_size=batch_size)
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size)

    # print(next(iter(train_dataloader))[0].max())
    args.char2int = train_dataset.char2int
    args.model_size = 'small'
    trainer.fit(model=DistillModule(**vars(args)),
                train_dataloaders=train_dataloader, val_dataloaders=test_dataloader)

if __name__ == '__main__':
    main()