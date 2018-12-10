import json
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch
import re

st = "<s>"
ed = "</s>"
unk = "<unk>"
pad_tok = "<pad>"


def collate_fn(batch_data, pad_value):
    batch_data.sort(key=lambda x: len(x), reverse=True)
    batch_data = [torch.tensor(x) for x in batch_data]
    lens = [len(x) for x in batch_data]
    padded = pad_sequence(batch_data, batch_first=True, padding_value=pad_value)
    # packed = pack_padded_sequence(padded, lens, batch_first=True)
    return padded

def my_pad_sequence(batch, pad_value):
    max_len = max([len(b) for b in batch])
    batch = [b + [pad_value] * (max_len - len(b)) for b in batch]
    return torch.tensor(batch)

class BatchManager:
    def __init__(self, datas, batch_size):
        self.steps = int(len(datas) / batch_size)
        # comment following two lines to neglect the last batch
        # if self.steps * batch_size < len(datas):
        #    self.steps += 1
        self.datas = datas
        self.batch_size = batch_size
        self.bid = 0

    def next_batch(self):
        batch = list(self.datas[self.bid * self.batch_size: (self.bid + 1) * self.batch_size])
        # batch = collate_fn(batch, pad_value=3)
        batch = my_pad_sequence(batch, 3)
        self.bid += 1
        if self.bid == self.steps:
            self.bid = 0
        return batch


class myCollate:
    def __init__(self, pad_value=3):
        self.pad_value = pad_value
        
    def collate_fn(self, batch_data):
        batch_data.sort(key=lambda x: len(x), reverse=True)
        batch_data = [torch.tensor(x) for x in batch_data]
        lens = [len(x) for x in batch_data]
        padded = pad_sequence(batch_data, batch_first=True, padding_value=self.pad_value)
        # packed = pack_padded_sequence(padded, lens, batch_first=True)
        return padded
    
    def __call__(self, batch_data):
        return self.collate_fn(batch_data)


def build_vocab(filelist=['sumdata/train/train.article.txt', 'sumdata/train/train.title.txt'],
                vocab_file='sumdata/vocab.json', min_count=0):
    print("Building vocab with min_count=%d..." % min_count)
    freq = defaultdict(int)
    for file in filelist:
        fin = open(file, "r", encoding="utf8")
        for _, line in enumerate(fin):
            for word in line.strip().split():
                freq[word] += 1
        fin.close()
    print('Number of all words: %d' % len(freq))
    
    vocab = {'<s>': 0, '</s>': 1, 'UNK': 2, '<pad>': 3}
    if 'UNK' in freq:
        freq.pop('UNK')
    for word in freq:
        if freq[word] > min_count:
            vocab[word] = len(vocab)
    print('Number of filtered words: %d, %f%% ' % (len(vocab), len(vocab)/len(freq)*100))

    json.dump(vocab, open(vocab_file,'w'))
    return freq


def load_embedding_vocab(embedding_path):
    fin = open(embedding_path)
    vocab = set([])
    for _, line in enumerate(fin):
        vocab.add(line.split()[0])
    return vocab


def build_vocab_from_embeddings(embedding_path, data_file_list):
    embedding_vocab = load_embedding_vocab(embedding_path)
    vocab = {'<s>': 0, '</s>': 1, '<unk>': 2, '<pad>': 3}

    for file in data_file_list:
        fin = open(file)
        for _, line in enumerate(fin):
            for word in line.split():
                if (word in embedding_vocab) and (word not in vocab):
                    vocab[word] = len(vocab)
    return vocab


def load_data_with_padding(filename, vocab, max_len=100, n_data=None, st='<s>', ed='</s>', unk='<unk>'):

    fin = open(filename, "r", encoding="utf8")
    datas = []
    for idx, line in enumerate(fin):
        if idx == n_data or line == '':
            break

        sample = np.ones(max_len, dtype=np.int32) * vocab[ed]
        sample[0] = vocab[st]
        words = line.strip().split()
        for i in range(min(len(words), max_len-2)):
            sample[i+1] = vocab[words[i]] if words[i] in vocab else vocab[unk]

        datas.append(sample)

    return np.array(datas, np.long)


def load_data(filename, vocab, n_data=None, target=False):
    fin = open(filename, "r", encoding="utf8")
    datas = []
    for idx, line in enumerate(fin):
        if idx == n_data or line == '':
            break
        words = line.strip().split()
        # if target:
        words = ['<s>'] + words + ['</s>']
        sample = [vocab[w if w in vocab else unk] for w in words]
        datas.append(sample)
    return datas


class MyDatasets(Dataset):
    def __init__(self, filename, vocab, n_data=None):
        self.datas = load_data(filename, vocab, n_data)
        self._size = len(self.datas)
    
    def __getitem__(self, idx):
        return self.datas[idx]
    
    def __len__(self):
        return self._size


def getDataLoader(filepath, vocab, n_data, batch_size, num_workers=0):
    dataset = MyDatasets(filepath, vocab, n_data)
    loader = DataLoader(dataset, batch_size, num_workers=num_workers, collate_fn=myCollate(vocab[pad_tok]))
    return loader
