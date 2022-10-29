import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

from utils import *
from utils import PAD_TAG, START_TAG, STOP_TAG
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, pad_sequence
from distill_emb_model import DistillEmb, create_am_distill_emb

IMPOSSIBLE = -1e4


class BiLSTMCRF(nn.Module):
    def __init__(self, vocab_size,  tagset_size, input_size, hidden_size, 
                word2index, num_rnn_layers=1, rnn="lstm",
                 rnn_dropout=0.5, fc_dropout=0.5, emb_dropout=0.1):
        super(BiLSTMCRF, self).__init__()

        self.word2index = word2index
        self.input_size = input_size
        self.tagset_size = tagset_size

        self.embedding = nn.Embedding(vocab_size, input_size)

        RNN = nn.LSTM if rnn == "LSTM" else nn.GRU
        if num_rnn_layers == 1:
            self.lstm = RNN(input_size, hidden_size, bidirectional=True, batch_first=True)
        else:
            self.lstm = RNN(input_size, hidden_size, bidirectional=True, batch_first=True,
                 num_layers=num_rnn_layers, dropout=rnn_dropout)

        # self.fc1 = nn.Linear(hidden_size*2, hidden_size)

        self.emb_dropout = nn.Dropout2d(emb_dropout)
        self.fc_dropout = nn.Dropout(fc_dropout)
        self.norm0 = nn.LayerNorm(300)
        self.norm1 = nn.LayerNorm(hidden_size*2)

        self.crf = CRF(hidden_size*2, self.tagset_size)


    def __build_features(self, x, char_x):
        masks = x.gt(0)
        mask_idx = masks.sum(1)

        x = self.embedding(x)

        x = self.emb_dropout(x)
        x = self.norm0(x)
        packed_x = pack_padded_sequence(x, mask_idx.cpu().numpy(), batch_first=True, enforce_sorted=False)
        packed_x, (h, c) = self.lstm(packed_x)

        x, input_sizes = pad_packed_sequence(packed_x, batch_first=True)

        # x = torch.cat((h[0], h[1]), dim=1)
        x = self.fc_dropout(x)
        x = self.norm1(x)

        return x, masks

    def loss(self, xs, mask_idx, tags):
        features, masks = self.__build_features(xs, mask_idx)
        loss = self.crf.loss(features, tags, masks=masks)
        return loss

    def forward(self, xs, mask_idx):
        # Get the emission scores from the BiLSTM
        features, masks = self.__build_features(xs, mask_idx)
        scores, tag_seq = self.crf(features, masks)
        return scores, tag_seq

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


class DistillBiLSTMCRF(nn.Module):
    def __init__(self,  tagset_size, input_size, hidden_size, charset_path,
                 num_rnn_layers=1, rnn="lstm",
                 rnn_dropout=0.5, fc_dropout=0.5, emb_dropout=0.1):
        super(DistillBiLSTMCRF, self).__init__()

        self.input_size = input_size
        self.tagset_size = tagset_size

        RNN = nn.LSTM if rnn == "LSTM" else nn.GRU
        if num_rnn_layers == 1:
            self.lstm = RNN(input_size, hidden_size, bidirectional=True, batch_first=True)
        else:
            self.lstm = RNN(input_size, hidden_size, bidirectional=True, batch_first=True,
                 num_layers=num_rnn_layers, dropout=rnn_dropout)

        self.embedding = create_am_distill_emb(charset_path, emb_dropout)
        # self.fc1 = nn.Linear(hidden_size*2, hidden_size)

        self.emb_dropout = nn.Dropout2d(emb_dropout)
        self.fc_dropout = nn.Dropout(fc_dropout)
        self.norm0 = nn.LayerNorm(300)
        self.norm1 = nn.LayerNorm(hidden_size*2)

        self.crf = CRF(hidden_size*2, self.tagset_size)

    def __build_features(self, emb_x, x):
        xs = []
        masks = emb_x.gt(0)
        mask_idx = masks.sum(1)
        max_len = max(mask_idx)
        for i in range(len(x)):
            xx = self.embedding(x[i, :mask_idx[i]]).unsqueeze(0)
            xx = torch.relu(xx)
            xx = self.emb_dropout(xx)
            
            xx = self.norm0(xx).squeeze(0)

            remain_len = max_len - mask_idx[i]
            if remain_len > 0:
                xx = torch.cat([xx, torch.zeros((remain_len, self.embedding.output_size), device=xx.device, dtype=xx.dtype)])
            # print(x.shape, xx.shape)
            xs.append(xx)

        x = torch.stack(xs, dim=0)

        packed_x = pack_padded_sequence(x, mask_idx.cpu().numpy(), batch_first=True, enforce_sorted=False)
        packed_x, (h, c) = self.lstm(packed_x)
        x, input_sizes = pad_packed_sequence(packed_x, batch_first=True)

        x = self.fc_dropout(x)
        x = self.norm1(x)

        return x, masks

    def loss(self, xs, cs, tags):
        features, masks = self.__build_features(xs, cs)
        loss = self.crf.loss(features, tags, masks=masks)
        return loss

    def forward(self, xs, cs):
        # Get the emission scores from the BiLSTM
        features, masks = self.__build_features(xs, cs)
        scores, tag_seq = self.crf(features, masks)
        return scores, tag_seq


def log_sum_exp(x):
    """calculate log(sum(exp(x))) = max(x) + log(sum(exp(x - max(x))))
    """
    max_score = x.max(-1)[0]
    return max_score + (x - max_score.unsqueeze(-1)).exp().sum(-1).log()


class CRF(nn.Module):
    """General CRF module.
    The CRF module contain a inner Linear Layer which transform the input from features space to tag space.

    :param in_features: number of features for the input
    :param num_tag: number of tags. DO NOT include START, STOP tags, they are included internal.
    """

    def __init__(self, in_features, num_tags):
        super(CRF, self).__init__()

        self.num_tags = num_tags + 2
        self.start_idx = self.num_tags - 2
        self.stop_idx = self.num_tags - 1

        self.fc = nn.Linear(in_features, self.num_tags)

        # transition factor, Tij mean transition from j to i
        self.transitions = nn.Parameter(torch.randn(
            self.num_tags, self.num_tags), requires_grad=True)
        self.transitions.data[self.start_idx, :] = IMPOSSIBLE
        self.transitions.data[:, self.stop_idx] = IMPOSSIBLE

    def forward(self, features, masks):
        """decode tags

        :param features: [B, L, C], batch of unary scores
        :param masks: [B, L] masks
        :return: (best_score, best_paths)
            best_score: [B]
            best_paths: [B, L]
        """
        features = self.fc(features)
        return self.__viterbi_decode(features, masks[:, :features.size(1)].float())

    def loss(self, features, ys, masks):
        """negative log likelihood loss
        B: batch size, L: sequence length, D: dimension

        :param features: [B, L, D]
        :param ys: tags, [B, L]
        :param masks: masks for padding, [B, L]
        :return: loss
        """
        features = self.fc(features)

        L = features.size(1)
        masks_ = masks[:, :L].float()

        forward_score = self.__forward_algorithm(features, masks_)
        gold_score = self.__score_sentence(features, ys[:, :L].long(), masks_)
        loss = (forward_score - gold_score).mean()
        return loss

    def __score_sentence(self, features, tags, masks):
        """Gives the score of a provided tag sequence

        :param features: [B, L, C]
        :param tags: [B, L]
        :param masks: [B, L]
        :return: [B] score in the log space
        """
        B, L, C = features.shape

        # emission score
        emit_scores = features.gather(
            dim=2, index=tags.unsqueeze(-1)).squeeze(-1)

        # transition score
        start_tag = torch.full((B, 1), self.start_idx,
                               dtype=torch.long, device=tags.device)
        tags = torch.cat([start_tag, tags], dim=1)  # [B, L+1]
        trans_scores = self.transitions[tags[:, 1:], tags[:, :-1]]

        # last transition score to STOP tag
        last_tag = tags.gather(dim=1, index=masks.sum(
            1).long().unsqueeze(1)).squeeze(1)  # [B]
        last_score = self.transitions[self.stop_idx, last_tag]

        score = ((trans_scores + emit_scores) * masks).sum(1) + last_score
        return score

    def __viterbi_decode(self, features, masks):
        """decode to tags using viterbi algorithm

        :param features: [B, L, C], batch of unary scores
        :param masks: [B, L] masks
        :return: (best_score, best_paths)
            best_score: [B]
            best_paths: [B, L]
        """
        B, L, C = features.shape

        bps = torch.zeros(B, L, C, dtype=torch.long,
                          device=features.device)  # back pointers

        # Initialize the viterbi variables in log space
        max_score = torch.full(
            (B, C), IMPOSSIBLE, device=features.device)  # [B, C]
        max_score[:, self.start_idx] = 0

        for t in range(L):
            mask_t = masks[:, t].unsqueeze(1)  # [B, 1]
            emit_score_t = features[:, t]  # [B, C]

            # [B, 1, C] + [C, C]
            acc_score_t = max_score.unsqueeze(
                1) + self.transitions  # [B, C, C]
            acc_score_t, bps[:, t, :] = acc_score_t.max(dim=-1)
            acc_score_t += emit_score_t
            max_score = acc_score_t * mask_t + max_score * \
                (1 - mask_t)  # max_score or acc_score_t

        # Transition to STOP_TAG
        max_score += self.transitions[self.stop_idx]
        best_score, best_tag = max_score.max(dim=-1)

        # Follow the back pointers to decode the best path.
        best_paths = []
        bps = bps.cpu().numpy()
        for b in range(B):
            best_tag_b = best_tag[b].item()
            seq_len = int(masks[b, :].sum().item())

            best_path = [best_tag_b]
            for bps_t in reversed(bps[b, :seq_len]):
                best_tag_b = bps_t[best_tag_b]
                best_path.append(best_tag_b)
            # drop the last tag and reverse the left
            best_paths.append(best_path[-2::-1])

        return best_score, best_paths

    def __forward_algorithm(self, features, masks):
        """calculate the partition function with forward algorithm.
        TRICK: log_sum_exp([x1, x2, x3, x4, ...]) = log_sum_exp([log_sum_exp([x1, x2]), log_sum_exp([x3, x4]), ...])

        :param features: features. [B, L, C]
        :param masks: [B, L] masks
        :return:    [B], score in the log space
        """
        B, L, C = features.shape

        scores = torch.full((B, C), IMPOSSIBLE,
                            device=features.device)  # [B, C]
        scores[:, self.start_idx] = 0.
        trans = self.transitions.unsqueeze(0)  # [1, C, C]

        # Iterate through the sentence
        for t in range(L):
            emit_score_t = features[:, t].unsqueeze(2)  # [B, C, 1]
            # [B, 1, C] + [1, C, C] + [B, C, 1] => [B, C, C]
            score_t = scores.unsqueeze(1) + trans + emit_score_t
            score_t = log_sum_exp(score_t)  # [B, C]

            mask_t = masks[:, t].unsqueeze(1)  # [B, 1]
            scores = score_t * mask_t + scores * (1 - mask_t)
        scores = log_sum_exp(scores + self.transitions[self.stop_idx])
        return scores
