import torch
from torch import nn
import torch.nn.functional as F
from Beam import Beam


class DotAttention(nn.Module):
    """
    Dot attention calculation
    """
    def __init__(self):
        super(DotAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, enc_outs, s_prev):
        """
        calculate the context vector c_t, both the input and output are batch first
        :param enc_outs: the encoder states, in shape [batch, seq_len, dim]
        :param s_prev: the previous states of decoder, h_{t-1}, in shape [1, batch, dim]
        :return: c_t: context vector
        """
        alpha_t = torch.bmm(s_prev.transpose(0, 1), enc_outs.transpose(1, 2))  # [batch, 1, seq_len]
        alpha_t = self.softmax(alpha_t)
        c_t = torch.bmm(alpha_t, enc_outs)  # [batch, 1, dim]
        return c_t


class BahdanauAttention(nn.Module):
    def __init__(self, enc_dim, dec_dim):
        super(BahdanauAttention, self).__init__()
        self.W_U = nn.Linear((enc_dim + dec_dim), dec_dim)
        self.V = nn.Linear(dec_dim, 1)

    def forward(self, enc_outs, s_prev):
        """
        calculate the context vector c_t, both the input and output are batch first
        :param enc_outs: the encoder states, in shape [batch, seq_len, dim]
        :param s_prev: the previous states of decoder, h_{t-1}, in shape [1, batch, dim]
        :return: c_t: context vector
        """
        s_expanded = s_prev.transpose(0,1).expand(-1, enc_outs.shape[1], -1)
        cat = torch.cat([enc_outs, s_expanded], dim=-1)
        alpha_t = self.V(torch.tanh(self.W_U(cat))).transpose(1, 2) # [batch, 1, seq_len]
        e_t = F.softmax(alpha_t, dim=-1)
        c_t = torch.bmm(e_t, enc_outs)  # [batch, 1, dim]
        return c_t


class Model(nn.Module):
    def __init__(self, vocab, out_len=10, emb_dim=32, hid_dim=128):
        super(Model, self).__init__()
        self.out_len = out_len
        self.hid_dim = hid_dim
        self.emb_dim = emb_dim
        self.vocab = vocab

        self.softmax = nn.Softmax(dim=-1)

        self.embedding_look_up = nn.Embedding(len(self.vocab), self.emb_dim)

        # encoder (with selective gate)
        self.encoder = nn.GRU(self.emb_dim, self.hid_dim//2, batch_first=True, bidirectional=True)
        self.linear1 = nn.Linear(hid_dim, hid_dim)
        self.linear2 = nn.Linear(hid_dim, hid_dim)
        self.sigmoid = nn.Sigmoid()

        # self.attention_layer = DotAttention()
        self.attention_layer = BahdanauAttention(self.hid_dim, self.hid_dim)
        self.decoder = nn.GRU(self.emb_dim + self.hid_dim, self.hid_dim, batch_first=True)

        # maxout
        self.W = nn.Linear(emb_dim, 2 * hid_dim)
        self.U = nn.Linear(hid_dim, 2 * hid_dim)
        self.V = nn.Linear(hid_dim, 2 * hid_dim)

        self.dropout = nn.Dropout(p=0.5)

        self.decoder2vocab = nn.Linear(self.hid_dim, len(self.vocab))

        self.loss_layer = nn.CrossEntropyLoss(ignore_index=self.vocab['<pad>'])

    def forward(self, inputs, targets):
        outputs, hidden = self.encode(inputs)
        return outputs, hidden

    def encode(self, inputs):
        embeds = self.embedding_look_up(inputs)
        embeds = self.dropout(embeds)
        outputs, hidden = self.encoder(embeds)  # h_0 defaults to zero if not provided
        sn = torch.cat([hidden[0], hidden[1]], dim=-1).view(-1, 1, self.hid_dim)
        # [batch, seq_len, hid_dim] + [batch, 1, hid_dim] = [batch, seq_len, hid_dim]
        sGate = self.sigmoid(self.linear1(outputs) + self.linear2(sn))
        outputs = outputs * sGate
        return outputs, hidden

    def maxout(self, w, c_t, hidden):
        r_t = self.W(w) + self.U(c_t) + self.V(hidden.transpose(0,1))
        m_t = F.max_pool1d(r_t, kernel_size=2, stride=2)
        return m_t

    def decode(self, word, enc_outs, hidden):
        embeds = self.embedding_look_up(word).view(-1, 1, self.emb_dim)
        embeds = self.dropout(embeds)
        c_t = self.attention_layer(enc_outs, hidden)
        outputs, hidden = self.decoder(torch.cat([c_t, embeds], dim=-1), hidden)
        outputs = self.maxout(embeds, c_t, hidden).squeeze()  # comment this line to remove maxout
        logit = self.decoder2vocab(outputs).squeeze()
        return logit, hidden
