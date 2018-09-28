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
	def __init__(self, vocab_size, emb_dim, hid_dim, batch_size, vocab, beam_size=3, max_trg_len=25):
		super(Seq2SeqAttention, self).__init__()
		self.vocab_size = vocab_size
		self.emb_dim = emb_dim
		self.hid_dim = hid_dim
		self.beam_size = beam_size
		self.vocab = vocab
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

		# maxout
		self.W = nn.Linear(emb_dim, 2 *self.vocab_size)
		self.U = nn.Linear(hid_dim, 2 *self.vocab_size)
		self.V = nn.Linear(hid_dim, 2 *self.vocab_size)

		weight_mask = torch.ones(vocab_size).cpu()
		weight_mask[vocab['</s>']] = 0
		# self.loss_function = nn.CrossEntropyLoss(weight=weight_mask)
		# self.loss_function = nn.CrossEntropyLoss(ignore_index=self.vocab['</s>'])
		# using the above two loss function will lead to loss don't decrease, don't know why
		self.loss_layer = nn.CrossEntropyLoss()

	def init_hidden(self, batch_size):
		return torch.zeros(2, batch_size, self.hid_dim // 2).cuda()

	def forward(self, src_sentence, trg_sentence, test=False):
		enc_states, enc_hidden = self.encode(src_sentence)
		# res = self.greedyDecoder(enc_states, enc_hidden, test, trg_sentence)
		res = self.beamSearchDecoder(enc_states, enc_hidden, test, trg_sentence)
		return res  # logits or summaries

	def encode(self, sentence):
		"""
		Selective encoding
		:param sentence:  shape of [batch, seq_len]
		:return: states: encoder states, enc_hidden: the last state of encoder
		"""
		batch_size = sentence.shape[0]
		embeds = self.embedding_lookup(sentence) # [batch, seq_len, word_dim]
		enc_hidden = self.init_hidden(len(sentence)) # [2, batch, hid_dim/2]
		states, enc_hidden = self.biGRU(embeds, enc_hidden) # [batch, seq_len, hid_dim], [2, batch, hid_dim/2]
		s_n = torch.cat([enc_hidden[0,:,:], enc_hidden[1,:,:]], dim=-1).view(batch_size, 1, self.hid_dim)
		# [batch, seq_len, hid_dim] + [batch, 1, hid_dim] = [batch, seq_len, hid_dim]
		sGate = self.sigmoid(self.linear1(states) + self.linear2(s_n))
		states = states * sGate
		return states, enc_hidden

	def beamSearchDecoder(self, enc_states, hidden, test=False, sentence=None, st="<s>", ed="</s>", k=3):
		"""
		Decoder with beam search
		:param enc_states:
		:param hidden:
		:param test:
		:param sentence:
		:param st:
		:param ed:
		:param k:
		:return:
		"""
		batch_size = enc_states.shape[0]
		hidden = F.tanh(self.init_decoder_hidden(hidden[1])).view(1, batch_size, self.hid_dim)
		if test:
			beams = [Beam(k, self.vocab, hidden[:,i,:]).cuda() for i in range(batch_size)]

			for i in range(self.max_trg_len):
				for j in range(batch_size):
					logits, hidden = self.decoderStep(enc_states[j].view(1, -1, self.hid_dim).expand(k, -1, -1),
													  beams[j].get_hidden_state(),
													  beams[j].get_current_word())
					logLikelihood = torch.log(F.softmax(logits, dim=-1))
					beams[j].advance(logLikelihood, hidden)

			allHyp, allScores = [], []
			n_best = 1
			for b in range(batch_size):
				scores, ks = beams[b].sort_best()

				allScores += [scores[:n_best]]
				hyps = [beams[b].get_hyp(k) for k in ks[:n_best]]
				allHyp.append(hyps)

			return allHyp
			# return sentences
		else:
			max_seq_len = sentence.shape[1]
			logits = torch.zeros(batch_size, max_seq_len - 1, self.vocab_size).cuda()
			for i in range(max_seq_len - 1):
				# logit: [batch, 1, vocab_size]
				logit, hidden = self.decoderStep(enc_states, hidden, sentence[:, i])
				logits[:, i, :] = logit.squeeze()
			return logits

	def maxout(self, w, c, s):
		r_t = self.W(w) + self.U(c) + self.V(s)
		m_t = F.max_pool1d(r_t, kernel_size=2, stride=2)
		m_t = m_t.unsqueeze(1)
		return m_t

	def decoderStep(self, enc_states, hidden, word):
		embeds = self.embedding_lookup(word).view(-1, 1, self.emb_dim) # [batch, 1, dim]
		c_t = self.dotAttention(enc_states, hidden) # [batch, 1, dim]
		outputs, hidden = self.GRUdecoder(torch.cat([embeds, c_t], dim=-1), hidden.contiguous())
		m_t = self.maxout(embeds, c_t, outputs)
		logits = self.decoder2vocab(outputs)  # [batch, 1, vocab_size]
		return m_t, hidden

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
		batch_size = hidden.shape[1]
		# according to paper
		hidden = F.tanh(self.init_decoder_hidden(hidden[1])).view(1, batch_size, self.hid_dim)
		if test:
			word = torch.ones(batch_size, dtype=torch.long) * self.vocab[st]
			words = torch.zeros(batch_size, self.max_trg_len, dtype=torch.long)
			for i in range(self.max_trg_len-1):
				logit, hidden = self.decoderStep(enc_states, hidden, word)
				probs = F.softmax(logit, dim=-1)
				word = torch.argmax(probs, dim=-1).squeeze()
				words[:,i] = word
			words[:,-1] = torch.ones(batch_size, dtype=torch.long) * self.vocab[ed]
			return words
		else:
			max_seq_len = sentence.shape[1]
			logits = torch.zeros(batch_size, max_seq_len-1, self.vocab_size)
			for i in range(max_seq_len - 1):
				# logit: [batch, 1, vocab_size]
				logit, hidden = self.decoderStep(enc_states, hidden, sentence[:,i])
				logits[:,i,:] = logit.squeeze()
			return logits
