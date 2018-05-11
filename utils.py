import os
from collections import defaultdict
import json
import numpy as np
from torch.utils.data import Dataset, DataLoader
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


def load_data(fileX='PART_III.article', fileY='PART_III.summary', n_data=None, data_dir='./data/',
            vocab_file='vocab.json', xlen=100, ylen=25, st='<s>', ed = '</s>', unk='<unk>'):
    
    if not os.path.exists(data_dir + vocab_file):
        build_vocab()
    vocab = json.load(open(data_dir + vocab_file))
    f1 = open(data_dir + fileX)
    f2 = open(data_dir + fileY)
    X = []
    Y = []

    print('Max length for summary/ariticle: %d/%d' % (xlen, ylen))

    for idx, article in enumerate(f1):
        if idx == n_data:
            break
        summary = f2.readline()

        if article=='' or summary=='':
            break
        
        x = [vocab[st]] + [vocab[ed]]*(xlen-1)
        words = article.strip().split()
        for i in range(min(len(words), xlen-2)):
            x[i+1] = vocab[words[i]] if words[i] in vocab else vocab[unk]
        
        y = [vocab[st]] + [vocab[ed]]*(ylen-1)
        words = summary.strip().split()
        for i in range(min(len(words), ylen-2)):
            y[i+1] = vocab[words[i]] if words[i] in vocab else vocab[unk]
        
        X.append(x)
        Y.append(y)

    return torch.tensor(X), torch.tensor(Y), vocab


class MyDatasets(Dataset):
    def __init__(self, fileX, fileY, n_data=None, data_dir='data/',
                vocab_file='vocab.json', xlen=100, ylen=25, st='<s>', ed = '</s>', unk='<unk>'):
        self._size = n_data
        self._xlen = xlen
        self._ylen = ylen
        self.X, self.Y, self.vocab = load_data(fileX, fileY, n_data, data_dir, vocab_file, xlen, ylen, st, ed, unk)
        self.vocab_size = len(self.vocab)
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
    
    def __len__(self):
        return self._size


