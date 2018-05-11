import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class SelectiveBiGRU(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim):
        super(SelectiveBiGRU, self).__init__()
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.hidden = self.init_hidden()

        # self.word_embeddings = nn.Embedding.from_pretrained(vocab_size, emb_dim)
        self.word_embeddings = nn.Embedding(vocab_size, emb_dim)
        self.biGRU = nn.GRU(emb_dim, hid_dim, bidirectional=True)

        self.linear = nn.Linear(2*hid_dim, 2*hid_dim)
        self.sigmoid = nn.Sigmoid()

    def init_hidden(self):
        return torch.zeros(2, 1, self.hid_dim)
    
    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        states, self.hidden = self.biGRU(embeds.view(len(sentence), 1, -1), self.hidden)
        sGate = self.sigmoid(self.linear(states)) \
                + self.sigmoid(self.linear(self.hidden.view(1, -1, 2*self.hid_dim)))
        states = states * sGate
        return states, self.hidden

class AttentionDecoder(nn.Module):
    '''
    NOTICE:
    1. encoder_hidden_dim * 2 = decoder_hidden_dim, since encoder is bi-directional
    2. We use Bahdanaum attention, here, instead of Luong attention
    '''
    def __init__(self, vocab_size, emb_dim, hid_dim):
        super(AttentionDecoder, self).__init__()
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.hidden = self.init_hidden()
        self.loss = None

        self.word_embeddings = nn.Embedding(vocab_size, emb_dim)
        self.GRUdecoder = nn.GRU(emb_dim + hid_dim, hid_dim)
        self.linear = nn.Linear(hid_dim, vocab_size)
        self.softmax = nn.Softmax(dim=-1)
        
        self.attn_layer1 = nn.Linear(self.hid_dim, 1) # since encoder is bi-directional
        self.attn_layer2 = nn.Linear(self.emb_dim, 1)

    def init_hidden(self):
        return torch.zeros(1, 1, self.hid_dim)

    def forward(self, sentence, encoder_hidden, encoder_states, batch_size=1, test=False):
        if test:
            # TODO: beam search
            pass
        else:
            self.hidden = encoder_hidden.view(1,-1, self.hid_dim)
            attn_1 = self.attn_layer1(encoder_states)
            self.loss = torch.zeros(1,batch_size,1)
            for i in range(len(sentence)-1):
                embeds = self.word_embeddings(sentence[i]).view(1,-1, self.emb_dim)
                attn_2 = self.attn_layer2(embeds)
                attn_weights = self.softmax(attn_1 + attn_2) # 7*1 + 1; should be added for each elem in 7s
                c_t = torch.sum(attn_weights * encoder_states, dim=0).view(1,-1, self.hid_dim) #
                dec_out, self.hidden = self.GRUdecoder(torch.cat((embeds, c_t), dim=-1), self.hidden)
                probs = self.softmax(self.linear(self.hidden))
                self.loss += -torch.log(probs[:,:,sentence[i+1]])
            return torch.sum(self.loss)

