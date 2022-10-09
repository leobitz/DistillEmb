import torch
import numpy as np
from torch.utils.data import Dataset
import lib
import random

UNK_WORD = "<###>"

class DistillDataset(Dataset):
    
    def __init__(self, words, vocab, vocab2index, w2v_vectors, ft_vectors, charset_path,
         neg_seq_len=32,  max_word_len=13, pad_char=' '):
        self.words = words
        self.vocab = vocab
        self.vocab2index = vocab2index
        self.fvectors = ft_vectors
        self.wvectors = w2v_vectors
        
        self.pad_char = pad_char
        self.neg_seq_len = neg_seq_len

        self.max_word_len = max_word_len
        self.char2int, self.int2char = lib.build_charset(charset_path, space_index=0)


    def __len__(self):
        return len(self.vocab)
    
    def __getitem__(self, idx):
        
        target_word = self.vocab[idx]
        target_vec = self.wvectors[target_word] # word2vec embedding
        target_vec = np.concatenate([target_vec, self.fvectors[target_word]]) # concat with fasttext embedding

        neg_word = target_word
        while target_word == neg_word:
            neg_start = random.randint(0, len(self.words) - self.neg_seq_len)
            neg_end = neg_start + self.neg_seq_len

            neg_words = set(self.words[neg_start:neg_end]) # incase there are duplicate words
            neg_words = neg_words - set([target_word]) # incase the target word is in the negative word
            neg_words = list(neg_words)

            vecs = np.stack([self.wvectors[wrd] for wrd in neg_words]) # collect w2v embedding of negatives
            fvecs = np.stack([self.fvectors[wrd] for wrd in neg_words]) # collect fasttext embedding of negatives
            vecs = np.concatenate([vecs, fvecs], axis=1) # concat them

            xsims = np.linalg.norm(vecs - target_vec, axis=1) # L2 norm (similarity)

            sims = np.argsort(xsims) # index of the most similar
            neg_word = neg_words[sims[0]] # index the word with the negative word with the highest similarity to the postive

        target_chars = lib.word2ids(self.char2int, target_word, self.pad_char, self.max_word_len)
        target_chars = torch.LongTensor(target_chars)
        
        pos_w2v = torch.Tensor(self.wvectors[target_word])
        pos_ft = torch.Tensor(self.fvectors[target_word])
        neg_w2v = torch.Tensor(self.wvectors[neg_word])
        neg_ft = torch.Tensor(self.fvectors[neg_word])

        return target_chars, pos_w2v, pos_ft, neg_w2v, neg_ft



class ClassificationDataset(Dataset):

    def __init__(self, data_rows,  word2index, label2index, charset_path, 
            max_seq_len=100,  word_output=False, max_word_len=13, pad_char=' ', exclude_classes=set([])):
        self.char2int, self.int2char = lib.build_charset(charset_path, space_index=0)
        
        self.max_word_len = max_word_len
        self.pad_char = pad_char

        self.word_output = word_output

        self.data = []
        self.labels = []
        self.max_seq_len = 0
        for ir, row in enumerate(data_rows):
            label = row[0]
            text = row[1]
            if label in exclude_classes:
                continue
            line = text.strip().split(' ')
            if len(line) > self.max_seq_len:
                self.max_seq_len = len(line)
            self.data.append(line)
            self.labels.append(label)

        # if data max length is greater than what the user specified, use the user-specified max length
        if max_seq_len < self.max_seq_len:
            self.max_seq_len = max_seq_len

        self.class_labels = tuple(sorted(set(self.labels)))

        self.word2index = word2index
        self.label2index = label2index


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        inputs = self.data[idx]
        label = self.labels[idx]

        if len(inputs) > self.max_seq_len:
            inputs = inputs[:self.max_seq_len]

        if self.word_output:
            inputs = [self.word2index[x] if x in self.word2index else self.word2index[UNK_WORD] for x in inputs]
            inputs = inputs + [self.word2index[UNK_WORD]] * (self.max_seq_len - len(inputs))
        else:
            inputs = lib.sen_word_to_word_ids(inputs, self.char2int, self.pad_char, self.max_word_len)

        return inputs, self.label2index[label], len(inputs)

def collate_fun(batch):

    batch_words, batch_labels, batch_mask_idx = [], [], []
    # print(batch_labels)
    for (_words, _labels, _mask_id) in batch:
        batch_words.append(torch.LongTensor(_words))
        batch_labels.append(_labels)
        batch_mask_idx.append(_mask_id)
    
    batch_mask_idx = torch.LongTensor(batch_mask_idx)
    if len(batch_words[0].shape) == 1: # for word-index outputs only
        batch_words = torch.stack(batch_words)
    batch_labels = torch.LongTensor(batch_labels)

    return batch_words, batch_labels, batch_mask_idx