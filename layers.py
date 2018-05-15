import torch
from torch import nn


class Seq2SeqAttention(nn.Module):
	def __init__(self, vocab_size, emb_dim, hid_dim, batch_size, device, max_trg_len=25):
		super(Seq2SeqAttention, self).__init__()
		self.vocab_size = vocab_size
		self.emb_dim = emb_dim
		self.hid_dim = hid_dim
		self.batch_size = batch_size
		self.enc_hidden = None
		self.dec_hidden = None
		self.device = torch.device('cpu') if device is None else device
		self.max_trg_len = max_trg_len

		self.embedding_lookup = nn.Embedding(vocab_size, emb_dim)

		# encoder
		self.biGRU = nn.GRU(emb_dim, hid_dim//2, bidirectional=True)
		self.linear1 = nn.Linear(hid_dim, hid_dim)
		self.linear2 = nn.Linear(hid_dim, hid_dim)
		self.sigmoid = nn.Sigmoid()

		# decoder
		self.GRUdecoder = nn.GRU(emb_dim + hid_dim, hid_dim)
		self.linear_vocab = nn.Linear(hid_dim, vocab_size)
		self.softmax = nn.Softmax(dim=-1)
		self.attn_layer1 = nn.Linear(self.hid_dim, 1)
		self.attn_layer2 = nn.Linear(self.emb_dim, 1)

	def init_hidden(self, batch_size):
		return torch.zeros(2, batch_size, self.hid_dim // 2, device=self.device)

	def forward(self, src_sentence, trg_sentence, test=False):
		enc_states, enc_hidden = self.encode(src_sentence)
		loss = self.decode(enc_states, enc_hidden, test, trg_sentence)
		return loss

	def encode(self, sentence):
		embeds = self.embedding_lookup(sentence)
		self.enc_hidden = self.init_hidden(len(sentence))
		states, self.enc_hidden = self.biGRU(embeds.view(-1, len(sentence), self.emb_dim), self.enc_hidden)
		sGate = self.sigmoid(self.linear1(states) + self.linear2(self.enc_hidden.view(1, -1, self.hid_dim)))
		states = states * sGate
		return states, self.enc_hidden

	def decode(self, enc_states, enc_hidden, test=False, sentence=None):
		if test:
			# TODO: beam search
			pass
		else:
			self.hidden = enc_hidden.view(1, -1, self.hid_dim)
			attn_1 = self.attn_layer1(enc_states)
			loss = torch.zeros(1, device=self.device)
			for i in range(self.max_trg_len - 1):
				embeds = self.embedding_lookup(sentence[:, i]).view(1, -1, self.emb_dim)
				attn_2 = self.attn_layer2(embeds)
				# len*batch*dim + 1*batch*dim; should be added for each elem in 7s
				attn_weights = self.softmax((attn_1 + attn_2).view(len(sentence), -1))
				c_t = torch.sum(attn_weights.view(-1, len(sentence), 1) * enc_states, dim=0).view(1, -1, self.hid_dim)  #
				dec_out, self.dec_hidden = self.GRUdecoder(torch.cat((embeds, c_t), dim=-1), self.hidden)
				probs = self.softmax(self.linear_vocab(self.hidden))
				for j in range(len(sentence)):
					loss += -torch.log(probs[:, j, sentence[j, i + 1]])
			return loss / len(sentence)

