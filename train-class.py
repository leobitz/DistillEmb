
from argparse import ArgumentParser
import random

import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from torch.utils.data import DataLoader
import sys
from class_model import  create_model
from distill_dataset import ClassificationDataset, UNK_WORD, distill_collate_fun, emb_collate_fun
import lib
from pytorch_lightning.callbacks import ModelCheckpoint
import os
import numpy as np
import warnings
warnings.filterwarnings("ignore", ".*does not have many workers.*")

class ClassifyModule(pl.LightningModule):

    def __init__(self, class_indexices, word2index, **kwargs) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=['class_indexices', "word2index", "test_models", "test_trail_ids"])
        self.model = create_model(self.hparams, word2index)

        self.criterion = nn.CrossEntropyLoss() if self.hparams.num_classes > 1 else nn.BCELoss()
        self.class_indexices = class_indexices

    def training_step(self, batch, batch_idx):

        x, y, seq_len = batch
        preds = self.model(x, seq_len)

        if self.hparams.num_classes == 1:
            sig_pred = torch.sigmoid(preds.view(-1))
            preds = torch.round(sig_pred.detach().cpu()).numpy()
            loss = self.criterion(sig_pred, y.float())
        else:
            loss = self.criterion(preds, y)
            preds = torch.argmax(preds.detach().cpu(), dim=1).numpy()

        targets = y.detach().cpu().numpy()
        acc, pre, rec, f1 = self._report(preds, targets, self.class_indexices)

        self.log("train_loss", loss, batch_size=len(x))
        self.log("train_f1", f1, batch_size=len(x))
        self.log("train_precision", pre, batch_size=len(x))
        self.log("train_recall", rec, batch_size=len(x))
        self.log("train_acc", acc, batch_size=len(x))

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
        
        if self.hparams.num_classes == 1:
            sig_pred = torch.sigmoid(preds.view(-1))
            preds = torch.round(sig_pred.detach().cpu()).numpy()
            loss = self.criterion(sig_pred, y.float())
        else:
            loss = self.criterion(preds, y)
            preds = torch.argmax(preds.detach().cpu(), dim=1).numpy()

        targets = y.detach().cpu().numpy()

        acc, pre, rec, f1 = self._report(
            preds, targets, class_indexices=self.class_indexices)

        self.log("val_loss", loss, batch_size=len(x))
        self.log("val_f1", f1, batch_size=len(x))
        self.log("val_precision", pre, batch_size=len(x))
        self.log("val_recall", rec, batch_size=len(x))
        self.log("val_acc", acc, batch_size=len(x))

        return {"loss": loss, "f1": f1}

    def test_step(self, batch, batch_idx):

        x, y, seq_len = batch
        preds = self.model(x, seq_len)

        if self.hparams.num_classes == 1:
            sig_pred = torch.sigmoid(preds.view(-1))
            preds = torch.round(sig_pred.detach().cpu()).long().numpy()
            loss = self.criterion(sig_pred, y.float())
        else:
            loss = self.criterion(preds, y)
            preds = torch.argmax(preds.detach().cpu(), dim=1).numpy()

        
        targets = y.detach().cpu().numpy()
        # print(targets, preds)
        acc, pre, rec, f1 = self._report(
            preds, targets, class_indexices=self.class_indexices)

        self.log("test_loss", loss, batch_size=len(x))
        self.log("test_f1", f1, batch_size=len(x))
        self.log("test_precision", pre, batch_size=len(x))
        self.log("test_recall", rec, batch_size=len(x))
        self.log("test_acc", acc, batch_size=len(x))

        return {"loss": loss, "f1": f1}

    def _report(self, Y_preds, Y_test, class_indexices):
        if self.hparams.num_classes > 1:
            average='macro'
        else:
            average='binary'
        pre = precision_score(
            Y_test, Y_preds, labels=class_indexices, average=average, zero_division=0)
        rec = recall_score(Y_test, Y_preds, labels=class_indexices,
                           average=average, zero_division=0)
        f1 = f1_score(Y_test, Y_preds, labels=class_indexices,
                      average=average, zero_division=0)
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
        parser.add_argument('--model-size', type=str,
                            default='small', help='model width: small (512) and large(768)')
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
parser.add_argument('--dataset-folder', type=str, default="tig",
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

parser = ClassifyModule.add_model_specific_args(parser)
parser = pl.Trainer.add_argparse_args(parser)
args = parser.parse_args()

if args.emb_type != 'CNN':
    if args.vocab_file == None:
        sys.exit('No vocab file provided')
    if args.vector_file == None or not os.path.exists(args.vector_file):
        sys.exit("Embedding file doesn't exist")
else:
    if args.vector_file == None:
        sys.exit('DistillEmb model doesnt exits')

args.test_models = not args.train_model

train_file = f"{args.dataset_folder}/clean-train.csv"
test_file = f"{args.dataset_folder}/clean-test.csv"
dev_file = f"{args.dataset_folder}/clean-dev.csv"

train_data = pd.read_csv(train_file).to_numpy()
# np.random.shuffle(train_data)
new_data_size = int(len(train_data) * args.data_size)
train_data = train_data[:new_data_size]

test_data = pd.read_csv(test_file).to_numpy()
dev_data = pd.read_csv(dev_file).to_numpy()

class_labels = sorted(set(train_data[:, 0]))
print("class_labels", class_labels)


word2index = None
label2index = {v: k for k, v in enumerate(class_labels)}
if args.vocab_file != None and args.emb_type != "CNN":
    task_tokens = open(args.vocab_file, encoding='utf-8').read().split()
    vocab = sorted(set(task_tokens))

    task_tokens = []
    for line in train_data[:, 1].flatten():
        tokens = line.split(" ")
        task_tokens.extend(tokens)

    current_vocab = set(task_tokens)
    if UNK_WORD in current_vocab:
        current_vocab.remove(UNK_WORD)

    vocab = [x for x in vocab if x in current_vocab]

    vocab.insert(0, UNK_WORD)

    word2index = {v: k for k, v in enumerate(vocab)}
    args.vocab_size = len(word2index)

train_dataset = ClassificationDataset(data_rows=train_data, word2index=word2index, label2index=label2index,
                                      charset_path=args.charset_path, pad_char=' ', 
                                      max_seq_len=args.max_seq_len, word_output=(args.emb_type != "CNN"))

test_dataset = ClassificationDataset(data_rows=test_data, word2index=word2index, label2index=label2index,
                                     charset_path=args.charset_path, pad_char=' ', 
                                    max_seq_len=args.max_seq_len, word_output=(args.emb_type != "CNN"))
dev_dataset = ClassificationDataset(data_rows=dev_data, word2index=word2index, label2index=label2index,
                                     charset_path=args.charset_path, pad_char=' ', 
                                    max_seq_len=args.max_seq_len, word_output=(args.emb_type != "CNN"))
if args.emb_type == 'CNN':
    collate_fun = distill_collate_fun
else:
    collate_fun = emb_collate_fun

train_dataloader = DataLoader(
    train_dataset,   shuffle=True, collate_fn=collate_fun, num_workers=0, batch_size=args.batch_size, drop_last=True)
test_dataloader = DataLoader(
    test_dataset,  shuffle=False, collate_fn=collate_fun, num_workers=0, batch_size=args.batch_size, drop_last=False)
dev_dataloader = DataLoader(
    dev_dataset,  shuffle=False, collate_fn=collate_fun, num_workers=0, batch_size=args.batch_size, drop_last=False)


args.num_classes = len(class_labels) if len(class_labels) > 2 else 1


checkpoint_cb = ModelCheckpoint(
    save_top_k=-1,
    every_n_epochs=1,
    dirpath=f'saves/{args.exp_name}/{args.trial_id}',
    filename='{epoch}-{val_loss:.5f}-{val_f1:.5f}')

m = ClassifyModule(list(label2index.values()), word2index, **vars(args))

if args.emb_type != "CNN":
    word2vec = lib.load_word_embeddings(args.vector_file, target_words=word2index, header=False)
    n_loaded = m.model.init_emb(w2v=word2vec)
    print("Loaded embs in %", n_loaded * 100 / len(word2index))
    

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

    trainer = pl.Trainer.from_argparse_args(args, logger=logger, accumulate_grad_batches=args.grad_accumulate, callbacks=[checkpoint_cb])

    trainer.fit(model=m,
                train_dataloaders=train_dataloader,
                val_dataloaders=dev_dataloader)

else:
    trainer = pl.Trainer.from_argparse_args(args)

    ids = args.test_trial_ids.split("-")
    trail_accs = []
    max_model_name = None
    max_val_acc = 0
    max_val_index = 0
    for idx in ids:
        folder_name = f'saves/{args.exp_name}/{idx}'
        names = sorted(os.listdir(folder_name), key=lambda x: int(x.split('-')[0].split('=')[1]))
        accs = [float(x[-12:-5]) for x in names]
        max_index = np.argmax(accs)
        if accs[max_index] > max_val_acc:
            max_val_acc = accs[max_index]
            max_val_index = max_index
            max_model_name = f"{folder_name}/{names[max_index]}"
        trail_accs.append(accs)
    trail_accs = np.array(trail_accs)
    mean_trail_accs = trail_accs.mean(axis=0)
    max_epoch = mean_trail_accs.argmax()
    args.word2index = word2index
    args.class_indexices = list(label2index.values())
    test_accs = []
    final_names = []
    for idx in ids:
        folder_name = f'saves/{args.exp_name}/{idx}'
        names = sorted(os.listdir(folder_name), key=lambda x: int(x.split('-')[0].split('=')[1]))
        name = names[max_epoch]
        path = f"{folder_name}/{name}"
        final_names.append(path)
        print(path)
        m = ClassifyModule.load_from_checkpoint(path, **vars(args))
        m.eval()
        m.freeze()
        result = trainer.test(m, dataloaders=test_dataloader)
        test_accs.append([result[0][x] for x in sorted(result[0].keys())])

    train_vocab = set([])
    for row in train_data:
        train_vocab.update(row[1].split())

    dev_vocab = set([])
    for row in dev_data:
        dev_vocab.update(row[1].split())

    test_vocab = set([])
    for row in test_data:
        test_vocab.update(row[1].split())

    tr_ts = train_vocab.intersection(test_vocab)
    tr_dv = train_vocab.intersection(dev_vocab)

    ts_ratio = len(tr_ts) * 100 / len(test_vocab)
    dv_ratio = len(tr_dv) * 100 / len(dev_vocab)

    print("=========================== Test RESULT ==========================")
    print("Max Epoch", max_epoch)
    header = "exp-name,test,test_acc,test_f1,test_loss,test_precision,test_recall,Epoch,Test-Train,Dev-Train"
    ave = np.mean(test_accs, axis=0)
    std = np.std(test_accs, axis=0)
    test_line = [args.exp_name, "test" ] + list(ave) + [max_epoch, ts_ratio, dv_ratio] + [-1] + list(std)
    print(sorted(result[0].keys()))
    print(ave)
    m = ClassifyModule.load_from_checkpoint(max_model_name, **vars(args))
    m.eval()
    m.freeze()
    max_test_result = trainer.test(m, dataloaders=test_dataloader)
    max_test_line = [args.exp_name, "max-test" ] + [max_test_result[0][x] for x in sorted(max_test_result[0].keys())] + [max_val_index, ts_ratio, dv_ratio] + [-1] + list(std)
    print("Max Test Result", [max_test_result[0][x] for x in sorted(max_test_result[0].keys())])
    print("Max file: ", max_model_name)
    print("=========================== Test END ==============================")
    
    val_accs = []
    for name in final_names:
        m = ClassifyModule.load_from_checkpoint(name, **vars(args))
        m.eval()
        m.freeze()
        result = trainer.test(m, dataloaders=dev_dataloader)
        val_accs.append([result[0][x] for x in sorted(result[0].keys())])

    print("=========================== Validation Result  ====================")
    ave = np.mean(val_accs, axis=0)
    std = np.std(val_accs, axis=0)
    print(sorted(result[0].keys()))
    val_line = [args.exp_name, "val" ] + list(ave) + [max_epoch, ts_ratio, dv_ratio] + [-1] + list(std)
    print(ave)
    print(max_model_name)
    m = ClassifyModule.load_from_checkpoint(max_model_name, **vars(args))
    m.eval()
    m.freeze()
    max_test_result = trainer.test(m, dataloaders=dev_dataloader)
    max_val_line = [args.exp_name, "max-val" ] + [max_test_result[0][x] for x in sorted(max_test_result[0].keys())] + [max_val_index, ts_ratio, dv_ratio] + [-1] + list(std)
    print("Max Val Result", [max_test_result[0][x] for x in sorted(max_test_result[0].keys())])
    print("=========================== Validation END   ========================")

    all_text = ""
    lines = [test_line, max_test_line, val_line, max_val_line]
    for line in lines:
        line = ",".join([str(x) for x in line])
        line = line  + "\n"
        all_text += line
    
    dsize = args.exp_name.split("-")[-1]
    index = args.exp_name.index(dsize)
    exp_name  = args.exp_name[:index-1]

    with open("saves/"+exp_name, mode='a', encoding='utf-8') as f:
        f.write(all_text)
    # print(f'Test-Train Vocab Ratio: {ts_ratio}, Dev-Train Vocab Ratio: {dv_ratio}')
