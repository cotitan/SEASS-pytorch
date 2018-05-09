import torch
from torch import nn
import torch.nn.functional as F

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
        gru_out, self.hidden = self.biGRU(embeds.view(len(sentence), 1, -1), self.hidden)
        sGate = self.sigmoid(self.linear(gru_out)) \
                + self.sigmoid(self.linear(self.hidden.view(1, -1, 2*self.hid_dim)))
        gru_out = gru_out * sGate
        return gru_out, self.hidden

# NOTICE: encoder_hidden_dim * 2 = decoder_hidden_dim, since encoder is bi-directional

class AttentionDecoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim):
        super(AttentionDecoder, self).__init__()
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.hidden = self.init_hidden()
        self.encoder_hidden = encoder_hidden
        self.encoder_out = encoder_out

        self.word_embeddings = nn.Embedding(vocab_size, emb_dim)
        self.GRUdecoder = nn.GRU(emb_dim + hid_dim, hid_dim)
        self.linear = nn.Linear(hid_dim, vocab_size)
        self.softmax = nn.Softmax()
        
        self.attn_layer1 = nn.Linear(self.hid_dim, 1) # since encoder is bi-directional
        self.attn_layer2 = nn.Linear(self.hid_dim, 1)


    def init_hidden(self):
        return torch.zeros(1, 1, self.hid_dim)

    def forward(self, sentence, encoder_hidden, encoder_out, test=False):
        if test:
            # TODO: beam search
            pass
        else:
            self.hidden = encoder_out
            attn_1 = self.attn_layer1(encoder_hidden)
            loss = 0
            for i in range(len(sentence)-1):
                embeds = self.word_embeddings(sentence[i])
                attn_2 = self.attn_layer2(embeds)
                attn_weights = self.softmax(attn_1 + attn_2) # 7*1 + 1; should be added for each elem in 7s
                c_t = torch.sum(attn_weights * encoder_hidden, axis=0) #
                dec_out, self.hidden = self.GRUdecoder(torch.cat(embeds, c_t), self.hidden)
                probs = self.linear(self.hidden[0])
                loss += -torch.log(probs[sentence[i+1]])
            return loss

def test():
    encoder = SelectiveBiGRU(10, 20, 30)
    x = torch.tensor([0,1,2,3,4,5,6], dtype=torch.long)
    # gru_out.size()=(len(x), batch_size, hid_dim*2)
    # hidden.size() = [2, batch_size, hid_dim*2]; here 2 means 2 layers
    gru_out, hidden = encoder(x)
    decoder = 
    print(gru_out.shape, hidden.shape)

if __name__ == '__main__':
    test()