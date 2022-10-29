from os.path import join, exists
import numpy as np
from tqdm import tqdm
import torch
from utils import *
import lib

FILE_VOCAB = "vocab.json"
FILE_TAGS = "tags.json"
FILE_DATASET = "dataset.txt"
FILE_DATASET_CACHE = "{}_cache_{}.npz"

class Preprocessor:
    def __init__(self, config_dir, save_config_dir=None, verbose=False):
        self.config_dir = config_dir
        self.verbose = verbose

        self.vocab, self.vocab_dict = self.__load_list_file(FILE_VOCAB, offset=1, verbose=verbose)
        self.tags, self.tags_dict = self.__load_list_file(FILE_TAGS, verbose=verbose)
        if save_config_dir:
            self.__save_config(save_config_dir)

        self.PAD_IDX = 0
        self.OOV_IDX = len(self.vocab)
        self.__adjust_vocab()

    def __load_list_file(self, file_name, offset=0, verbose=False):
        file_path = join(self.config_dir, file_name)
        if not exists(file_path):
            raise ValueError('"{}" file does not exist.'.format(file_path))
        else:
            elements = load_json_file(file_path)
            elements_dict = {w: idx + offset for idx, w in enumerate(elements)}
            if verbose:
                print("config {} loaded".format(file_path))
            return elements, elements_dict

    def __adjust_vocab(self):
        self.vocab.insert(0, PAD)
        self.vocab_dict[PAD] = 0

        self.vocab.append(OOV)
        self.vocab_dict[OOV] = len(self.vocab) - 1

    def __save_config(self, dst_dir):
        char_file = join(dst_dir, FILE_VOCAB)
        save_json_file(self.vocab, char_file)

        tag_file = join(dst_dir, FILE_TAGS)
        save_json_file(self.tags, tag_file)

        if self.verbose:
            print("tag dict file => {}".format(tag_file))
            print("tag dict file => {}".format(char_file))

    @staticmethod
    def __cache_file_path(corpus_dir, max_seq_len, dataset_file):
        return join(corpus_dir, FILE_DATASET_CACHE.format(dataset_file, max_seq_len))
    
    def _load_dataset(self, corpus_dir, max_seq_len, dataset_file, max_word_len, pad_char, charset_path):
        ds_path = self.__cache_file_path(corpus_dir, max_seq_len, dataset_file)
        if not exists(ds_path):
            xs, cs, ys, ms = self.__build_corpus(corpus_dir=corpus_dir, max_seq_len=max_seq_len, charset_path=charset_path, 
                                        max_word_len=max_word_len, pad_char=pad_char, dataset_file=dataset_file)
        else:
            print("loading dataset {} ...".format(ds_path))
            dataset = np.load(ds_path)
            xs, cs, ys, ms = dataset["xs"], dataset["cs"], dataset["ys"], dataset["ms"]

        xs, cs, ys, ms = map(
            torch.tensor, (xs, cs, ys, ms)
        )

        return xs, cs, ys, ms

    def load_dataset(self, corpus_dir, max_seq_len, max_word_len, pad_char, charset_path):
        """load the train set

        :return: (xs, ys)
            xs: [B, L]
            ys: [B, L, C]
        """
        train_xs, train_cs, train_ys, train_ms = self._load_dataset(corpus_dir, max_seq_len, f'train.txt', max_word_len, pad_char, charset_path)
        val_xs, val_cs, val_ys, val_ms = self._load_dataset(corpus_dir, max_seq_len, f'dev.txt', max_word_len, pad_char, charset_path)
        test_xs, test_cs, test_ys, test_ms = self._load_dataset(corpus_dir, max_seq_len, f'test.txt', max_word_len, pad_char, charset_path)

        return (train_xs, train_cs, train_ys, train_ms), (val_xs,val_cs, val_ys, val_ms), (test_xs, test_cs, test_ys, test_ms)

    def decode_tags(self, batch_tags):
        batch_tags = [
            [self.tags[t] for t in tags]
            for tags in batch_tags
        ]
        return batch_tags

    def sent_to_vector(self, sentence, max_seq_len=0):
        max_seq_len = max_seq_len if max_seq_len > 0 else len(sentence)
        vec = [self.vocab_dict.get(c, self.OOV_IDX) for c in sentence[:max_seq_len]]
        return vec + [self.PAD_IDX] * (max_seq_len - len(vec))

    def tags_to_vector(self, tags, max_seq_len=0):
        max_seq_len = max_seq_len if max_seq_len > 0 else len(tags)
        vec = [self.tags_dict[c] for c in tags[:max_seq_len]]
        return vec + [0] * (max_seq_len - len(vec))

    def __build_corpus(self, corpus_dir, max_seq_len, charset_path, max_word_len=13, pad_char="_", dataset_file='dataset.txt'):
        
        char2int, int2char = lib.build_charset(charset_path)
        pad_word = [char2int[pad_char]] * max_word_len
        file_path = join(corpus_dir, dataset_file)

        xs, ys, cs, ms = [], [], [], []
        with open(file_path, encoding="utf-8") as f:
            for idx, line in tqdm(enumerate(f), desc="parsing {}".format(file_path)):
                fields = line.strip().split("\t")
                if len(fields) != 2:
                    raise ValueError("format error in line {}, tabs count: {}".format(idx + 1, len(fields) - 1))

                sentence, tags = fields
                sentence = sentence.split()
                try:
                    if sentence[0] == "[":
                        sentence = json.loads(sentence)
                    tags = json.loads(tags)
                    sen = sentence[:max_seq_len] if len(sentence) > max_seq_len else sentence
                    ms.append(len(sen))
                    char_sen = lib.sen_word_to_word_ids(sen, char2int, pad_char=pad_char, max_word_len=max_word_len)
                    char_sen = char_sen + [pad_word] * (max_seq_len - len(char_sen))
                    cs.append(np.array(char_sen))
                    xs.append(self.sent_to_vector(sentence, max_seq_len=max_seq_len))
                    ys.append(self.tags_to_vector(tags, max_seq_len=max_seq_len))
                    if len(sentence) != len(tags):
                        raise ValueError('"sentence length({})" != "tags length({})" in line {}"'.format(
                            len(sentence), len(tags), idx + 1))
                except Exception as e:
                    raise ValueError("exception raised when parsing line {}\n\t{}\n\t{}".format(idx + 1, line, e))

        xs, ys = np.asarray(xs), np.asarray(ys)
        ms = np.asarray(ms, dtype=np.int64)
        cs = np.stack(cs)

        # save train set
        cache_file = self.__cache_file_path(corpus_dir, max_seq_len, dataset_file)
        np.savez(cache_file, xs=xs, ys=ys, cs=cs, ms=ms)
        return xs, cs, ys, ms
