import json
import numpy as np
from collections import defaultdict
import word2vec
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch
import threading


start_tok = "<s>"
end_tok = "</s>"
unk_tok = "<unk>"
pad_tok = "<pad>"
""" Caution:
In training data, unk_tok='<unk>', but in test data, unk_tok='UNK'.
This is reasonable, because if the unk_tok you predict is the same as the
unk_tok in the test data, then your prediction would be regard as correct,
but since unk_tok is unknown, it's impossible to give a correct prediction
"""


def my_pad_sequence(batch, pad_value):
    max_len = max([len(b) for b in batch])
    batch = [b + [pad_value] * (max_len - len(b)) for b in batch]
    return torch.tensor(batch)


class BatchManager:
    def __init__(self, datas, batch_size):
        self.steps = int(len(datas) / batch_size)
        # comment following two lines to neglect the last batch
        if self.steps * batch_size < len(datas):
           self.steps += 1
        self.datas = datas
        self.batch_size = batch_size
        self.bid = 0

    def next_batch(self):
        batch = list(self.datas[self.bid * self.batch_size: (self.bid + 1) * self.batch_size])
        batch = my_pad_sequence(batch, 0) # pad_index
        self.bid += 1
        if self.bid == self.steps:
            self.bid = 0
        return batch


def build_vocab(filelist=['sumdata/train/train.article.txt', 'sumdata/train/train.title.txt'],
                vocab_file='sumdata/vocab.json', min_count=0, n_vocab=130000):
    print("Building vocab with min_count=%d..." % min_count)
    freq = defaultdict(int)
    for file in filelist:
        fin = open(file, "r", encoding="utf8")
        for _, line in enumerate(fin):
            for word in line.strip().split():
                freq[word] += 1
        fin.close()
    print('Number of all words: %d' % len(freq))
    
    vocab = {pad_tok: 0, start_tok: 1, end_tok: 2, unk_tok: 3}
    if unk_tok in freq:
        freq.pop(unk_tok)
    for word in freq:
        if freq[word] > min_count:
            vocab[word] = len(vocab)
        if len(vocab) >= n_vocab:
            break
    print('Number of filtered words: %d, %f%% ' % (len(vocab), len(vocab)/len(freq)*100))

    json.dump(vocab, open(vocab_file,'w'))
    return freq


def load_embedding_vocab(embedding_path):
    fin = open(embedding_path)
    vocab = set([])
    for _, line in enumerate(fin):
        vocab.add(line.split()[0])
    return vocab


def load_word2vec_embedding(filepath):
    w2v = word2vec.load(filepath)
    weights = w2v.vectors
    vocab = {}

    if start_tok not in w2v.vocab:
        w2v.vocab = np.concatenate([np.array([start_tok]), w2v.vocab])
        weights = np.concatenate([np.zeros((1, weights.shape[1])), weights], axis=0)

    if pad_tok not in w2v.vocab:
        vocab[pad_tok] = 0
        weights = np.concatenate([np.zeros((1, weights.shape[1])), weights], axis=0)
    for tok in w2v.vocab:
        vocab[tok] = len(vocab)
    return vocab, torch.tensor(weights, dtype=torch.float)


def build_vocab_from_embeddings(embedding_path, data_file_list):
    embedding_vocab = load_embedding_vocab(embedding_path)
    vocab = {start_tok: 0, end_tok: 1, unk_tok: 2, pad_tok: 3}

    for file in data_file_list:
        fin = open(file)
        for _, line in enumerate(fin):
            for word in line.split():
                if (word in embedding_vocab) and (word not in vocab):
                    vocab[word] = len(vocab)
    return vocab


def load_data(filename, vocab, n_data=None, target=False):
    fin = open(filename, "r", encoding="utf8")
    datas = []
    for idx, line in enumerate(fin):
        if idx == n_data or line == '':
            break
        words = line.strip().split()
        # if target:
        words = ['<s>'] + words + ['</s>']
        sample = [vocab[w if w in vocab else unk_tok] for w in words]
        datas.append(sample)
    return datas
