import torch
from torch import nn
import torch.nn.functional as F


class DotAttention(nn.Module):
	"""
	Dot attention calculation
	"""
	def __init__(self):
		super(DotAttention, self).__init__()

	def forward(self, enc_states, h_prev):
		"""
		calculate the context vector c_t, both the input and output are batch first
		:param enc_states: the encoder states, in shape [batch, seq_len, dim]
		:param h_prev: the previous states of decoder, h_{t-1}, in shape [1, batch, dim]
		:return: c_t: context vector
		"""
		alpha_t = torch.bmm(h_prev.transpose(0,1), enc_states.transpose(1,2)) # [batch, 1, seq_len]
		alpha_t = F.softmax(alpha_t, dim=-1)
		c_t = torch.bmm(alpha_t, enc_states) # [batch, 1, dim]
		return c_t


class AttentionGRUCell(nn.Module):
	def __init__(self, vocab_size, emb_dim, hid_dim, mlp_hid_dim=100):
		super(AttentionGRUCell, self).__init__()
		self.hid_dim = hid_dim
		self.mlp_hid_dim = mlp_hid_dim
		self.dec_hid_weights = nn.Linear(hid_dim, mlp_hid_dim)
		self.enc_hid_weights = nn.Linear(hid_dim, mlp_hid_dim)
		self.mlp_out = nn.Linear(mlp_hid_dim, 1)
		self.GRUCell = nn.GRUCell(hid_dim + emb_dim, hid_dim)
		self.decoder2vocab = nn.Linear(hid_dim, vocab_size)

	def forward(self, hidden_prev, word_emb, encoder_states):
		# (1,batch,1)+(len,batch,1)=(len,batch,1)
		e_t = self.mlp_out(
			F.tanh(self.dec_hid_weights(hidden_prev) + self.enc_hid_weights(encoder_states)))
		attn_weights = F.softmax(e_t, dim=0)
		c_t = torch.sum(attn_weights * encoder_states, dim=0)
		inputs = torch.cat([c_t, word_emb], dim=-1)
		hidden = self.GRUCell(inputs, hidden_prev)
		logits = self.decoder2vocab(hidden)
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
		self.biGRU = nn.GRU(emb_dim, hid_dim//2, batch_first=True, bidirectional=True)
		self.linear1 = nn.Linear(hid_dim, hid_dim)
		self.linear2 = nn.Linear(hid_dim, hid_dim)
		self.sigmoid = nn.Sigmoid()

		# decoder,
		self.decoderCell = AttentionGRUCell(self.vocab_size, self.emb_dim, self.hid_dim)
		self.GRUdecoder = nn.GRU(emb_dim + hid_dim, hid_dim, batch_first=True)
		self.dotAttention = DotAttention()
		self.decoder2vocab = nn.Linear(hid_dim, vocab_size)
		self.init_decoder_hidden = nn.Linear(hid_dim//2, hid_dim)

		weight_mask = torch.ones(vocab_size).cpu()
		weight_mask[vocab['</s>']] = 0
		# self.loss_function = nn.CrossEntropyLoss(weight=weight_mask)
		# self.loss_function = nn.CrossEntropyLoss(ignore_index=self.vocab['</s>'])
		# using the above two loss function will lead to loss don't decrease, don't know why
		self.loss_function = nn.CrossEntropyLoss()

	def init_hidden(self, batch_size):
		return torch.zeros(2, batch_size, self.hid_dim // 2, device=self.device)

	def forward(self, src_sentence, trg_sentence, test=False):
		enc_states, enc_hidden = self.encode(src_sentence)
		res = self.greedyDecoder(enc_states, enc_hidden, test, trg_sentence)
		return res  # logits or summaries

	def encode(self, sentence):
		"""
		Selective encoding
		:param sentence:  shape of [batch, seq_len]
		:return: states: encoder states, enc_hidden: the last state of encoder
		"""
		embeds = self.embedding_lookup(sentence) # [batch, seq_len, word_dim]
		enc_hidden = self.init_hidden(len(sentence)) # [2, batch, hid_dim/2]
		states, enc_hidden = self.biGRU(embeds, enc_hidden) # [batch, seq_len, hid_dim], [2, batch, hid_dim/2]
		s_n = torch.cat([enc_hidden[0,:,:], enc_hidden[1,:,:]], dim=-1).view(self.batch_size, 1, self.hid_dim)
		# [batch, seq_len, hid_dim] + [batch, 1, hid_dim] = [batch, seq_len, hid_dim]
		sGate = self.sigmoid(self.linear1(states) + self.linear2(s_n))
		states = states * sGate
		return states, enc_hidden

	def beam_search(self, hidden_prev, probs):
		# hidden_prev: beam_size*batch*dim
		# probs: batch * beam_size * vocab_size
		values, indices = torch.topk(probs.view(self.batch_size, -1), k=self.beam_size, largest=False)
		acc_probs = values
		hidden_ids = indices / self.vocab_size
		words = indices % self.vocab_size
		hidden = torch.zeros(self.beam_size, self.batch_size, self.hid_dim, device=self.device)
		for i in range(self.beam_size):
			for j in range(self.batch_size):
				hidden[i,j,:] = hidden_prev[hidden_ids[j, i], j, :]
		return hidden, acc_probs, words

	def decoderStep(self, enc_states, hidden, word):
		embeds = self.embedding_lookup(word).view(self.batch_size, 1, self.emb_dim) # [batch, 1, dim]
		c_t = self.dotAttention(enc_states, hidden) # [batch, 1, dim]
		outputs, hidden = self.GRUdecoder(torch.cat([embeds, c_t], dim=-1), hidden)
		logits = self.decoder2vocab(outputs) # [batch, 1, vocab_size]
		return logits, hidden

	def greedyDecoder(self, enc_states, hidden, test=False, sentence=None, st='<s>', ed='</s>'):
		"""
		Decoder with greedy search
		:param enc_states:
		:param hidden: shape = [2, batch, hid_dim/2]
		:param test: boolean value, test or train
		:param sentence:
		:param st: start token
		:param ed:
		:return: logits
		"""
		# according to paper
		hidden = F.tanh(self.init_decoder_hidden(hidden[1])).view(1, self.batch_size, self.hid_dim)
		if test:
			word = torch.ones(self.batch_size) * self.vocab[st]
			words = [word]
			for i in range(self.max_trg_len-1):
				logit, hidden = self.decoderStep(enc_states, hidden, word)
				word = torch.argmax(logit, dim=-1).squeeze()
				words.append(word)
			words.append(torch.ones(self.batch_size) * self.vocab[ed])
			return words
		else:
			max_seq_len = sentence.shape[1]
			logits = torch.zeros(self.batch_size, max_seq_len, self.vocab_size, device=self.device)
			for i in range(max_seq_len - 1):
				# logit: [batch, 1, vocab_size]
				logit, hidden = self.decoderStep(enc_states, hidden, sentence[:,i])
				logits[:,i+1,:] = logit.squeeze()
			return logits

	def decode(self, enc_states, enc_hidden, test=False, sentence=None, st='<s>', ed='</s>'):
		batch_size = enc_states.shape[1]
		beam_size = self.beam_size
		if test:
			# TODO: beam search
			with torch.no_grad():
				# 1st step, choose top K to gain the 1st word
				words = torch.tensor([self.vocab[st]]*batch_size, device=self.device)
				embeds = self.embedding_lookup(words)
				hidden, logits = self.decoderCell(enc_hidden.squeeze(), embeds, enc_states)
				probs = -torch.log(F.softmax(logits, dim=-1))
				hidden, acc_probs, words = self.beam_search(hidden.view(1, -1, self.hid_dim), probs)

				summaries = torch.zeros(batch_size, beam_size, self.max_trg_len, device=self.device)
				summaries[:, :, 0] = words

				dec_hidden = torch.zeros(beam_size, batch_size, self.hid_dim, device=self.device)
				probs = torch.zeros(batch_size, beam_size, self.vocab_size, device=self.device)

				for i in range(1, self.max_trg_len - 1):

					for j in range(beam_size):
						embeds = self.embedding_lookup(words[:,j])
						dec_hidden[j], logits = self.decoderCell(hidden[j], embeds, enc_states)
						mask = torch.ones(batch_size, 1, device=self.device)
						mask[words[:,j]==self.vocab['</s>']] = 0
						probs[:, j] = acc_probs[:, j].view(-1, 1) - torch.log(
								F.softmax(logits, dim=-1)) * mask

					hidden, acc_probs, words = self.beam_search(dec_hidden, probs)

					summaries[:, :, i] = words

				_, indices = torch.topk(acc_probs, k=1, largest=False)

				summary = torch.zeros(batch_size, self.max_trg_len, device=self.device)
				for i in range(batch_size):
					summary[i] = summaries[i, indices[i], :]
				return summary

		else:
			loss = torch.zeros(1, device=self.device)
			hidden_prev = enc_hidden.squeeze()
			for i in range(self.max_trg_len - 1):
				embeds = self.embedding_lookup(sentence[:, i])
				hidden_prev, logits = self.decoderCell(hidden_prev, embeds, enc_states)
				loss += self.loss_function(logits, sentence[:,i])
			return loss / len(sentence)
