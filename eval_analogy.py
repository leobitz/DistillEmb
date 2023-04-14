from gensim.test.utils import datapath
from gensim.models import KeyedVectors


def evaluate(vector_file, eval_file):

    wv = KeyedVectors.load_word2vec_format(vector_file, binary=False, unicode_errors='ignore')
    result = wv.evaluate_word_analogies(eval_file)
    
    actual_result = {} # dict for holding section
    for i in range(len(result[1])):
        section = result[1][i]["section"]
        correct = len(result[1][i]["correct"])
        incorrect = len(result[1][i]["incorrect"])
        total = correct + incorrect
        actual_result[section] = correct * 100.0 / total
    # print(actual_result)
    return actual_result

