import torch
import numpy as np
from torch.utils.data import Dataset
import lib
import random


class DistillDataset(Dataset):
    
    def __init__(self, words, vocab, vocab2index, w2v_vectors, ft_vectors, charset_path,
         neg_seq_len=32,  max_word_len=13, pad_char='_'):
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


