import torch
from torch import nn
import torch.nn.functional as F
from Beam import Beam

torch.manual_seed(1)


class LuongAttention(nn.Module):
    def __init__(self, enc_dim, dec_dim, align='concat'):
        super(LuongAttention, self).__init__()
        assert align in ['dot', 'general', 'concat']
        self.align = align
        if align == 'concat':
            self.W = nn.Linear(enc_dim + dec_dim, dec_dim)
            self.V = nn.Linear(dec_dim, 1)
        elif align == 'general':
            self.W = nn.Linear(enc_dim, dec_dim)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, enc_outs, ht, mask=None):
        """
        :param enc_outs: shape = [batch, seqlen, enc_dim]
        :param ht: shape = [1, batch, dec_dim]
        """
        if self.align == 'dot':
            scores = torch.bmm(ht.transpose(0,1), enc_outs.transpose(1,2))
        elif self.align == 'general':
            scores = torch.bmm(ht.transpose(0,1), enc_outs.transpose(1,2))
        else:
            ht_expand = ht.transpose(0,1).expand_as(enc_outs)
            cat = torch.cat([enc_outs, ht_expand], dim=-1)
            scores = self.V(torch.tanh(self.W(cat))).transpose(1, 2) # batch, 1, seqlen
        if mask is not None:
            scores.masked_fill(mask, -1e9)
        weights = self.softmax(scores)
        # batch,1,seqlen * batch,seqlen,enc_dim = batch, 1, enc_dim
        c_t = torch.bmm(weights, enc_outs)
        return c_t


class BahdanauAttention(nn.Module):
    def __init__(self, enc_dim, dec_dim):
        super(BahdanauAttention, self).__init__()
        self.W_U = nn.Linear((enc_dim + dec_dim), dec_dim)
        self.V = nn.Linear(dec_dim, 1)

    def forward(self, enc_outs, s_prev, mask=None):
        """
        calculate the context vector c_t, both the input and output are batch first
        :param enc_outs: the encoder states, in shape [batch, seq_len, dim]
        :param s_prev: the previous states of decoder, h_{t-1}, in shape [1, batch, dim]
        :param mask: mask for pad value
        :return: c_t: context vector
        """
        s_expanded = s_prev.transpose(0,1).expand(-1, enc_outs.shape[1], -1)
        cat = torch.cat([enc_outs, s_expanded], dim=-1)
        alpha_t = self.V(torch.tanh(self.W_U(cat))).transpose(1, 2) # [batch, 1, seq_len]
        if mask is not None:
            alpha_t = alpha_t.masked_fill(mask, -1e9)
        e_t = F.softmax(alpha_t, dim=-1)
        c_t = torch.bmm(e_t, enc_outs)  # [batch, 1, dim]
        return c_t


class Model(nn.Module):
    def __init__(self, vocab, emb_dim=32, hid_dim=128, embeddings=None, attn='bahdanau'):
        super(Model, self).__init__()
        assert attn in ['luong', 'bahdanau']
        self.hid_dim = hid_dim
        self.emb_dim = emb_dim
        self.vocab = vocab
        self.n_vocab = len(vocab)

        self.softmax = nn.Softmax(dim=-1)

        if embeddings is None:
            self.embedding_look_up = nn.Embedding(self.n_vocab, emb_dim, padding_idx=vocab['<pad>'])
        else:
            self.embedding_look_up = nn.Embedding.from_pretrained(embeddings, freeze=False)

        # encoder (with selective gate)
        self.encoder = nn.GRU(emb_dim, hid_dim//2, batch_first=True, bidirectional=True)
        self.linear1 = nn.Linear(hid_dim, hid_dim)
        self.linear2 = nn.Linear(hid_dim, hid_dim)
        self.sigmoid = nn.Sigmoid()

        self.attn = attn
        if attn == 'luong':
            # self.attn_layer = LuongAttention(hid_dim, hid_dim, align='dot')
            self.attn_layer = LuongAttention(hid_dim, hid_dim, align='concat')
            self.decoder = nn.GRU(emb_dim, hid_dim, batch_first=True)
            self.decoder2vocab = nn.Linear(hid_dim * 2, self.n_vocab)
        else:
            self.attn_layer = BahdanauAttention(hid_dim, hid_dim)
            self.decoder = nn.GRU(emb_dim + hid_dim, hid_dim, batch_first=True)
            self.decoder2vocab = nn.Linear(hid_dim, self.n_vocab)

        self.enc2dec = nn.Linear(hid_dim//2, hid_dim)

        # maxout
        self.W = nn.Linear(emb_dim, 2 * hid_dim)
        self.U = nn.Linear(hid_dim, 2 * hid_dim)
        self.V = nn.Linear(hid_dim, 2 * hid_dim)

        self.dropout = nn.Dropout(p=0.5)

        self.loss_layer = nn.CrossEntropyLoss(ignore_index=vocab['<pad>'])

    def init_decoder_hidden(self, hidden):
        hidden = torch.tanh(self.enc2dec(hidden[1]).unsqueeze(0))
        return hidden

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
        return self.dropout(m_t)

    def decode(self, word, enc_outs, hidden, mask=None):
        embeds = self.embedding_look_up(word).view(-1, 1, self.emb_dim)
        embeds = self.dropout(embeds)
        if self.attn == 'luong':
            outputs, hidden = self.decoder(embeds, hidden)
            c_t = self.attn_layer(enc_outs, hidden, mask)
        else:
            c_t = self.attn_layer(enc_outs, hidden, mask)
            outputs, hidden = self.decoder(torch.cat([c_t, embeds], dim=-1), hidden)
        outputs = self.maxout(embeds, c_t, hidden).squeeze()  # comment this line to remove maxout
        logit = self.decoder2vocab(outputs).squeeze()
        return logit, hidden
