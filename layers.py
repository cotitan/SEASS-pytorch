import torch
from torch import nn


class Seq2SeqAttention(nn.Module):
	def __init__(self, vocab_size, emb_dim, hid_dim, batch_size, vocab, device, beam_size=3, max_trg_len=25):
		super(Seq2SeqAttention, self).__init__()
		self.vocab_size = vocab_size
		self.emb_dim = emb_dim
		self.hid_dim = hid_dim
		self.batch_size = batch_size
		self.beam_size = beam_size
		self.enc_hidden = None
		self.dec_hidden = None
		self.vocab = vocab
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
		self.attn_layer2 = nn.Linear(self.hid_dim, 1)

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

	def decode(self, enc_states, enc_hidden, test=False, sentence=None, st='<s>', ed='</s>'):
		batch_size = enc_states.shape[1]
		if test:
			# TODO: beam search
			with torch.no_grad():
				self.dec_hidden = enc_hidden.view(1, -1, self.hid_dim)
				attn_1 = self.attn_layer1(enc_states)
				beam_size = self.beam_size

				h_prev = torch.cat([self.dec_hidden]*beam_size)  # shape=(beam_size, batch_size, hid_dim)
				word_prev = torch.tensor([[self.vocab[st]]*beam_size]*batch_size, device=self.device)
				acc_prob = torch.zeros(batch_size, beam_size, device=self.device)
				summaries = torch.zeros(batch_size, beam_size, self.max_trg_len, device=self.device)

				summaries[:, :, 0] = word_prev

				for i in range(self.max_trg_len - 1):
					dec_hidden = torch.zeros(batch_size, beam_size, self.hid_dim, device=self.device)

					probs = torch.zeros(batch_size, beam_size, self.vocab_size, device=self.device)
					for j in range(beam_size):
						embeds = self.embedding_lookup(word_prev[:, j]).view(1, -1, self.emb_dim)
						attn_2 = self.attn_layer2(h_prev[j])
						# len*batch*dim + 1*batch*dim; should be added for each elem in
						attn_weights = self.softmax((attn_1 + attn_2).view(batch_size, -1))
						c_t = torch.sum(attn_weights.view(-1, batch_size, 1) * enc_states, dim=0).view(1, -1, self.hid_dim)  #

						dec_states, dec_hidden[:, j] = self.GRUdecoder(
								torch.cat([embeds, c_t], dim=-1), h_prev[j].view(1, -1, self.hid_dim))
						probs[:, j] = acc_prob[:, j].view(-1, 1) - torch.log(
										self.softmax(self.linear_vocab(dec_states.view(-1, self.hid_dim))))

					values, indices = torch.topk(probs.view(batch_size, -1), k=beam_size, largest=False)
					acc_prob = values
					summaries_cur = torch.tensor(summaries, device=self.device)
					for j in range(beam_size):
						for k in range(batch_size):
							h_prev[j,k,:] = dec_hidden[k, indices[k,j]/self.vocab_size, :]
						word_prev[:, j] = indices[:, j] % self.vocab_size
						for k in range(batch_size):
							summaries_cur[k, j] = summaries[k, indices[k,j]/self.vocab_size, :]
							summaries_cur[k, j, i+1] = word_prev[k, j]
					summaries = summaries_cur
				_, indices = torch.topk(acc_prob, k=1, largest=False)

				summary = torch.zeros(batch_size, self.max_trg_len, device=self.device)
				for i in range(batch_size):
					summary[i] = summaries[i, indices[i], :]
				return summary

		else:
			self.dec_hidden = enc_hidden.view(1, -1, self.hid_dim)
			attn_1 = self.attn_layer1(enc_states)
			loss = torch.zeros(1, device=self.device)
			for i in range(self.max_trg_len - 1):
				embeds = self.embedding_lookup(sentence[:, i]).view(1, -1, self.emb_dim)
				attn_2 = self.attn_layer2(self.dec_hidden)
				# len*batch*dim + 1*batch*dim; should be added for each elem in 7s
				attn_weights = self.softmax((attn_1 + attn_2).view(len(sentence), -1))
				c_t = torch.sum(attn_weights.view(-1, len(sentence), 1) * enc_states, dim=0).view(1, -1, self.hid_dim)  #
				dec_states, self.dec_hidden = self.GRUdecoder(torch.cat((embeds, c_t), dim=-1), self.dec_hidden)
				probs = self.softmax(self.linear_vocab(dec_states))
				for j in range(len(sentence)):
					loss += -torch.log(probs[:, j, sentence[j, i + 1]])
			return loss / len(sentence) / self.max_trg_len

