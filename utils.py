import os
import json
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader


def build_vocab(filelist=['data/PART_I.article', 'data/PART_I.summary'],
                vocab_file='data/vocab.json', low_freq_bound=5):
    print("Building vocab...")
    freq = defaultdict(int)
    for file in filelist:
        fin = open(file, "r", encoding="utf8")
        for _, line in enumerate(fin):
            for word in line.strip().split():
                freq[word] += 1
        fin.close()
    print('Number of all words: %d' % len(freq))
    
    vocab = {'<s>': 0, '</s>': 1, 'UNK': 2}
    if 'UNK' in vocab:
        vocab.pop('UNK')
    for word in freq:
        if freq[word]>=low_freq_bound:
            vocab[word] = len(vocab)
    print('Number of filtered words: %d, %f%% ' % (len(vocab), len(vocab)/len(freq)*100))

    json.dump(vocab, open(vocab_file,'w'))


def load_data(filename, vocab, max_len=100, n_data=None, st='<s>', ed='</s>', unk='UNK'):

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


class MyDatasets(Dataset):
    def __init__(self, filename, vocab, max_len=100, n_data=None, st='<s>', ed='</s>', unk='UNK'):
        self._max_len = max_len
        self.datas = load_data(filename, vocab, max_len, n_data, st, ed, unk)
        self._size = len(self.datas)
    
    def __getitem__(self, idx):
        return self.datas[idx]
    
    def __len__(self):
        return self._size


def getDataLoader(filepath, vocab, max_len, n_data, batch_size, num_workers=1):
    dataset = MyDatasets(filepath, vocab, max_len, n_data)
    loader = DataLoader(dataset, batch_size, num_workers=num_workers)
    return loader
