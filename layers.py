import torch
from torch import nn
import torch.nn.functional as F

class AttentionGRUCell(nn.Module):
	def __init__(self, vocab_size, emb_dim, hid_dim):
		super(AttentionGRUCell, self).__init__()
		self.hid_dim = hid_dim
		self.dec_hid_weights = nn.Linear(hid_dim, 1)
		self.enc_hid_weights = nn.Linear(hid_dim, 1)
		self.GRUCell = nn.GRUCell(hid_dim + emb_dim, hid_dim)
		self.decoder2vocab = nn.Linear(hid_dim, vocab_size)

	def forward(self, hidden_prev, word_emb, encoder_states):
		# (1,batch,1)+(len,batch,1)=(len,batch,1)
		e_t = self.dec_hid_weights(hidden_prev) + self.enc_hid_weights(encoder_states)
		attn_weights = F.softmax(e_t, dim=0)
		c_t = torch.sum(attn_weights * encoder_states, dim=0).view(1, -1, self.hid_dim)
		inputs = torch.cat([c_t, word_emb], dim=-1)
		_, hidden = self.GRUCell(inputs, hidden_prev)
		logits = F.softmax(self.decoder2vocab(hidden), dim=-1)
		return hidden, logits


class Seq2SeqAttention(nn.Module):
	def __init__(self, vocab_size, emb_dim, hid_dim, batch_size, vocab, device, beam_size=3, max_trg_len=25):
		super(Seq2SeqAttention, self).__init__()
		self.vocab_size = vocab_size
		self.emb_dim = emb_dim
		self.hid_dim = hid_dim
		self.batch_size = batch_size
		self.beam_size = beam_size
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
		self.decoderCell = AttentionGRUCell(self.vocab_size, self.emb_dim, self.hid_dim)

		self.loss_function = nn.CrossEntropyLoss()

	def init_hidden(self, batch_size):
		return torch.zeros(2, batch_size, self.hid_dim // 2, device=self.device)

	def forward(self, src_sentence, trg_sentence, test=False):
		enc_states, enc_hidden = self.encode(src_sentence)
		res = self.decode(enc_states, enc_hidden, test, trg_sentence)
		return res  # loss or summaries

	def encode(self, sentence):
		embeds = self.embedding_lookup(sentence).transpose(0, 1)
		enc_hidden = self.init_hidden(len(sentence))
		states, enc_hidden = self.biGRU(embeds, enc_hidden)
		enc_hidden = torch.cat([enc_hidden[0], enc_hidden[1]], dim=-1).view(1, -1, self.hid_dim)
		sGate = self.sigmoid(self.linear1(states) + self.linear2(enc_hidden))
		states = states * sGate
		return states, enc_hidden

	def decode(self, enc_states, enc_hidden, test=False, sentence=None, st='<s>', ed='</s>'):
		batch_size = enc_states.shape[1]
		dec_hidden = enc_hidden
		if test:
			# TODO: beam search
			with torch.no_grad():
				attn_1 = self.attn_layer1(enc_states)
				beam_size = self.beam_size

				h_prev = torch.cat([dec_hidden]*beam_size)  # shape=(beam_size, batch_size, hid_dim)
				word_prev = torch.tensor([[self.vocab[st]]*beam_size]*batch_size, device=self.device)
				summaries = torch.zeros(batch_size, beam_size, self.max_trg_len, device=self.device)

				summaries[:, :, 0] = word_prev

				embeds = self.embedding_lookup(word_prev[:, 0]).view(1, -1, self.emb_dim)
				attn_2 = self.attn_layer2(dec_hidden)
				attn_weights = F.softmax(attn_1 + attn_2, dim=0)
				c_t = torch.sum(attn_weights * enc_states, dim=0).view(1, -1, self.hid_dim)
				dec_states, dec_hidden = self.GRUdecoder(torch.cat([embeds, c_t], dim=-1), dec_hidden)
				probs = - torch.log(F.softmax(self.linear_vocab(dec_states.view(-1, self.hid_dim)), dim=-1))

				values, indices = torch.topk(probs.view(batch_size, -1), k=beam_size, largest=False)
				acc_prob = values
				word_prev = indices

				summaries[:, :, 1] = word_prev

				for j in range(beam_size):
					h_prev[j] = dec_hidden

				for i in range(1, self.max_trg_len - 1):
					dec_hidden = torch.zeros(batch_size, beam_size, self.hid_dim, device=self.device)

					probs = torch.zeros(batch_size, beam_size, self.vocab_size, device=self.device)
					for j in range(beam_size):
						embeds = self.embedding_lookup(word_prev[:, j]).view(1, -1, self.emb_dim)
						attn_2 = self.attn_layer2(h_prev[j])
						# len*batch*1 + 1*batch*1
						attn_weights = F.softmax(attn_1 + attn_2, dim=0)
						c_t = torch.sum(attn_weights * enc_states, dim=0).view(1, -1, self.hid_dim)

						dec_states, dec_hidden[:, j] = self.GRUdecoder(
								torch.cat([embeds, c_t], dim=-1), h_prev[j].view(1, -1, self.hid_dim))
						probs[:, j] = acc_prob[:, j].view(-1, 1) - torch.log(
										F.softmax(self.linear_vocab(dec_states.view(-1, self.hid_dim)), dim=-1))

					values, indices = torch.topk(probs.view(batch_size, -1), k=beam_size, largest=False)
					acc_prob = values
					summaries_cur = torch.tensor(summaries, device=self.device)
					for j in range(beam_size):
						for k in range(batch_size):
							h_prev[j, k, :] = dec_hidden[k, indices[k, j]/self.vocab_size, :]
						word_prev[:, j] = indices[:, j] % self.vocab_size
						for k in range(batch_size):
							summaries_cur[k, j] = summaries[k, indices[k, j]/self.vocab_size, :]
							summaries_cur[k, j, i+1] = word_prev[k, j]
					summaries = summaries_cur
				_, indices = torch.topk(acc_prob, k=1, largest=False)

				summary = torch.zeros(batch_size, self.max_trg_len, device=self.device)
				for i in range(batch_size):
					summary[i] = summaries[i, indices[i], :]
				return summary

		else:
			loss = torch.zeros(1, device=self.device)
			hidden_prev = enc_hidden
			for i in range(self.max_trg_len - 1):
				embeds = self.embedding_lookup(sentence[:, i]).view(1, -1, self.emb_dim)
				hidden_prev, logits = self.decoderCell(hidden_prev, embeds, enc_states)
				loss += self.loss_function(logits, sentence[:,i])
			return loss / len(sentence)
