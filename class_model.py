import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, pad_sequence
import numpy as np

from distill_emb_model import DistillEmb, create_am_distill_emb

class LSTMTextClassifier(nn.Module):

    def __init__(self, vocab_size, input_size, hidden_size, n_outputs, word2index,  train_embeder=True,
             fc_dropout=0.6, emb_dropout=0.6, rnn_dropout=0.6, num_rnn_layers=1):
        super(LSTMTextClassifier, self).__init__()
        self.word2index = word2index
        self.input_size = input_size
        if num_rnn_layers == 1:
            self.lstm = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        else:
            self.lstm = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True,
                 num_layers=num_rnn_layers, dropout=rnn_dropout)
        self.fc1 = nn.Linear(hidden_size*2, n_outputs)

        self.embedding = nn.Embedding(vocab_size, input_size)
        self.embedding.requires_grad_(train_embeder)

        self.emb_dropout = nn.Dropout2d(emb_dropout)
        self.fc_dropout = nn.Dropout(fc_dropout)
        self.norm0 = nn.LayerNorm(300)
        self.norm1 = nn.LayerNorm(hidden_size*2)

    def forward(self, x, mask_idx):
        x = self.embedding(x)
        
        x = self.emb_dropout(x)
        x = self.norm0(x)
        packed_x = pack_padded_sequence(x, mask_idx.cpu().numpy(), batch_first=True, enforce_sorted=False)
        packed_x, (h, c) = self.lstm(packed_x)

        x, input_sizes = pad_packed_sequence(packed_x, batch_first=True)

        x = torch.cat((h[0], h[1]), dim=1)
        x = self.fc_dropout(x)
        x = self.norm1(x)

        x = self.fc1(x)
        return x


    def init_emb(self, fn=None, w2v=None):
        if fn != None:
            self.apply(fn)
        k = 0
        if w2v != None:
            vecs = np.random.uniform(-np.sqrt(0.06), np.sqrt(0.06), (len(self.word2index), self.input_size))
            for ik, (kw, vw) in enumerate(self.word2index.items()):
                if kw in w2v:
                    vecs[vw] = np.array(w2v[kw])
                    k += 1
            self.embedding.weight.data.copy_(torch.from_numpy(vecs))
        return k

    def init_model(self, fn=None, model_name=None):
        if fn != None:
            self.apply(fn)
        if model_name != None:
            checkpoint = torch.load(model_name)
            self.embeder.load_state_dict(checkpoint['model_state_dict'])


class CharLSTMTextClassifier(nn.Module):

    def __init__(self, input_size, hidden_size, n_outputs, charset_path, train_embeder=True,
             fc_dropout=0.6, emb_dropout=0.6, rnn_dropout=0.6, num_rnn_layers=1):
        super(CharLSTMTextClassifier, self).__init__()
        
        self.input_size = input_size
        if num_rnn_layers == 1:
            self.lstm = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        else:
            self.lstm = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True,
                 num_layers=num_rnn_layers, dropout=rnn_dropout)

        self.fc1 = nn.Linear(hidden_size*2, n_outputs)
        
        self.embedding = create_am_distill_emb(charset_path, emb_dropout)
        self.embedding.requires_grad_(train_embeder)
        
        
        self.emb_dropout = nn.Dropout2d(emb_dropout)
        self.fc_dropout = nn.Dropout(fc_dropout)
        self.norm0 = nn.LayerNorm(self.embedding.output_size)
        self.norm1 = nn.LayerNorm(hidden_size*2)


    def forward(self, x, mask_idx):
        xs = []

        max_len = mask_idx.max()
        for i in range(len(x)):
            xx = self.embedding(x[i]).unsqueeze(0)
            xx = self.emb_dropout(xx)
            xx = self.norm0(xx).squeeze(0)

            remain_len = max_len - mask_idx[i]
            if remain_len > 0:
                xx = torch.cat([xx, torch.zeros((remain_len, self.embedding.output_size), device=xx.device, dtype=xx.dtype)])
            
            xs.append(xx)

        x = torch.stack(xs, dim=0)

        packed_x = pack_padded_sequence(x, mask_idx.cpu().numpy(), batch_first=True, enforce_sorted=False)
        packed_x, (h, c) = self.lstm(packed_x)

        x = torch.cat((h[0], h[1]), dim=1)
        x = self.fc_dropout(x)
        x = self.norm1(x)

        x = self.fc1(x)
        return x

    def init_emb(self, model_name=None):
        if model_name != None:
            checkpoint = torch.load(model_name)
            self.embedding.load_state_dict(checkpoint['model_state_dict'])


def create_model(hparams, word2index):
    if hparams.emb_type != "CNN":
        model = LSTMTextClassifier(vocab_size=hparams.vocab_size,
                        input_size=hparams.embedding_dim,
                        hidden_size=hparams.hidden_dim,
                        n_outputs=hparams.num_classes,
                        word2index=word2index,
                        train_embeder=hparams.train_embedding,
                        fc_dropout=hparams.fc_dropout,
                        rnn_dropout=hparams.rnn_dropout,
                        emb_dropout=hparams.emb_dropout,
                        num_rnn_layers=hparams.num_rnn_layers)
    else:
        model = CharLSTMTextClassifier(
                        input_size=hparams.embedding_dim,
                        hidden_size=hparams.hidden_dim,
                        n_outputs=hparams.num_classes,
                        charset_path=hparams.charset_path,
                        train_embeder=hparams.train_embedding,
                        fc_dropout=hparams.fc_dropout,
                        rnn_dropout=hparams.rnn_dropout,
                        emb_dropout=hparams.emb_dropout,
                        num_rnn_layers=hparams.num_rnn_layers)

    return model