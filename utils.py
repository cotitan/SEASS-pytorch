import os
from collections import defaultdict
import json
import numpy as np
from torch.utils.data import Dataset
import torch

def build_vocab(filelist=['data/PART_I.article', 'data/PART_I.summary'],
                vocab_file='data/vocab.json', low_freq_bound=5):
    print("Building vocab...")
    freq = defaultdict(int)
    for file in filelist:
        fin = open(file)
        for _, line in enumerate(fin):
            for word in line.strip().split():
                freq[word] += 1
        fin.close()
    print('Number of all words: %d' % len(freq))
    
    vocab = {'<s>': 0, '</s>': 1, '<unk>': 2}
    for word in freq:
        if freq[word]>=low_freq_bound:
            vocab[word] = len(vocab)
    print('Number of filtered words: %d, %f%% ' % (len(vocab), len(vocab)/len(freq)*100))
    
    json.dump(vocab, open(vocab_file,'w'))


def load_data(filename, max_len=100, n_data=None, data_dir='./data/',
            vocab_file='vocab.json', st='<s>', ed = '</s>', unk='<unk>'):
    
    if not os.path.exists(data_dir + vocab_file):
        build_vocab()
    vocab = json.load(open(data_dir + vocab_file))

    fin = open(data_dir + filename)

    datas = []
    for idx, line in enumerate(fin):
        if idx == n_data or line == '':
            break

        sample = [vocab[st]] + [vocab[ed]]*(max_len-1)
        words = line.strip().split()
        for i in range(min(len(words), max_len-2)):
            sample[i+1] = vocab[words[i]] if words[i] in vocab else vocab[unk]
        
        datas.append(sample)

    return torch.tensor(datas), vocab


class MyDatasets(Dataset):
    def __init__(self, filename, max_len=100, n_data=None, data_dir='data/',
                vocab_file='vocab.json', st='<s>', ed = '</s>', unk='<unk>'):
        self._size = n_data
        self._max_len = max_len
        self.datas, self.vocab = load_data(filename, max_len, n_data, data_dir, vocab_file, st, ed, unk)
        self.vocab_size = len(self.vocab)
    
    def __getitem__(self, idx):
        return self.datas[idx]
    
    def __len__(self):
        return self._size


