import random


def build_charset(charset_file, space_index=-1):
    """
    returns character mapping to integer and vice versa
    it takes the tabular character arrangement

    space_index: index of the space character. Spaces is included only if space_index >= 0
    """
    charset = open(charset_file, encoding='utf-8').read().strip()
    charset = sorted([c for c in charset if c != ''])

    if space_index > -1:
        charset.insert(space_index, ' ')

    char2int = {c: x for x, c in enumerate(charset)}
    int2char = {x: c for x, c in enumerate(charset)}

    return char2int, int2char


def word2ids(char2int, word, pad_char, max_len):
        if len(word) > max_len:
            word = word[:max_len]

        ids =  [char2int[c] for c in word] 
        n_pad = (max_len - len(ids)) // 2
        pad = [char2int[pad_char]] * n_pad
        ids = pad + ids + pad
        ids = ids + [char2int[pad_char]] * (max_len - len(ids))
        return ids

def load_word_embeddings(file_path: str, target_words: set =None, header: bool =True, word_prob=1.0) -> dict:
    word2vec = {}
    with open(file_path, encoding='utf-8', errors='ignore') as f:
        if header:
            line = f.readline()
            n_vecs, dim = int(line.split(' ')[0]), int(line.split(' ')[1])
        for line in f:
            if random.random() < word_prob:
                line = line.strip().split(' ')
                word = line[0]
                vec = line[1:]
                if target_words == None or word in target_words:
                    word2vec[word] = [float(x) for x in vec]
                
    return word2vec

def ids2word(int2char, ids):
    return ''.join([int2char[i] for i in ids])

def ids2clean_word(int2char, ids):
    return ''.join([int2char[i] for i in ids if int2char[i] != 's'])

def sen_idx_to_word_ids(index2word, sen_idx,  pad_char,  max_word_len):
    return [word2ids(index2word[idx], 
                        pad_char,  max_word_len) for idx in sen_idx]

def sen_word_to_word_ids(words, char2int, pad_char,  max_word_len):
    return [word2ids(char2int, word, pad_char,  max_word_len) for word in words]


def load_corpus_words(path, line_prob=1.0):
    words = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            if random.random() < line_prob:
                words.extend(line.strip().split())

    return words
