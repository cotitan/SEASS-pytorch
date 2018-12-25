import os
import json
import torch
import argparse
import numpy as np
from utils import BatchManager, load_data
from Model import Model
from Beam import Beam
import torch.nn.functional as F

parser = argparse.ArgumentParser(description='Selective Encoding for Abstractive Sentence Summarization in pytorch')

parser.add_argument('--n_valid', type=int, default=189651,
					help='Number of validation data (up to 189651 in gigaword) [default: 189651])')
parser.add_argument('--input_file', type=str, default="sumdata/Giga/input.txt", help='input file')
parser.add_argument('--batch_size', type=int, default=64, help='Mini batch size [default: 32]')
parser.add_argument('--emb_dim', type=int, default=300, help='Embedding size [default: 256]')
parser.add_argument('--hid_dim', type=int, default=512, help='Hidden state size [default: 256]')
parser.add_argument('--maxout_dim', type=int, default=2, help='Maxout size [default: 2]')
parser.add_argument('--model_file', type=str, default='./models/params_0.pkl', help='model file path')
parser.add_argument('--search', type=str, default='greedy', help='greedy/beam')
args = parser.parse_args()
print(args)

if not os.path.exists(args.model_file):
	raise FileNotFoundError("model file not found")


def print_summaries(summaries, vocab):
	"""
	param summaries: in shape (seq_len, batch)
	"""
	i2w = {key: value for value, key in vocab.items()}

	for stnc in summaries:
		line = [i2w[tok] for tok in stnc if tok != vocab["</s>"]]
		print(" ".join(line))


def greedy(model, batch_x, max_trg_len=10):
	enc_outs, hidden = model.encode(batch_x)
	hidden = torch.cat([hidden[0], hidden[1]], dim=-1).unsqueeze(0)
	
	words = []
	word = torch.ones(hidden.shape[1], dtype=torch.long).cuda() * model.vocab["<s>"]
	for _ in range(max_trg_len):
		logit, hidden = model.decode(word, enc_outs, hidden)
		word = torch.argmax(logit, dim=-1)
		words.append(word.cpu().numpy())
	return np.array(words).T


def beam_search(model, batch_x, max_trg_len=10, k=3):
	enc_outs, hidden = model.encode(batch_x)
	hidden = torch.cat([hidden[0], hidden[1]], dim=-1).unsqueeze(0)

	beams = [Beam(k, model.vocab, hidden[:,i,:])
			for i in range(batch_x.shape[0])]
	
	for _ in range(max_trg_len):
		for j in range(len(beams)):
			hidden = beams[j].get_hidden_state()
			word = beams[j].get_current_word()
			enc_outs_j = enc_outs[j].unsqueeze(0).expand(k, -1, -1)
			logit, hidden = model.decode(word, enc_outs_j, hidden)
			# logit: [k x V], hidden: [k x hid_dim]
			probs = F.softmax(logit, -1)
			beams[j].advance(probs, hidden)

	allHyp, allScores = [], []
	n_best = 1
	for b in range(batch_x.shape[0]):
		scores, ks = beams[b].sort_best()
		allScores += [scores[:n_best]]
		hyps = [beams[b].get_hyp(k) for k in ks[:n_best]]
		allHyp.append(hyps)

	# shape of allHyp: [batch, 1, list]
	allHyp = [[int(w.cpu().numpy()) for w in hyp[0]] for hyp in allHyp]
	return allHyp


def my_test(valid_x, model):
	with torch.no_grad():
		for _ in range(valid_x.steps):
			batch_x = valid_x.next_batch().cuda()
			if args.search == "greedy":
				summaries = greedy(model, batch_x)
			elif args.search == "beam":
				summaries = beam_search(model, batch_x)
			else:
				print("Unknown search method")
			print_summaries(summaries, model.vocab)


def main():

	N_VALID = args.n_valid
	BATCH_SIZE = args.batch_size
	EMB_DIM = args.emb_dim
	HID_DIM = args.hid_dim

	vocab = json.load(open('sumdata/vocab.json'))
	valid_x = BatchManager(load_data(args.input_file, vocab, N_VALID), BATCH_SIZE)

	# model = Seq2SeqAttention(len(vocab), EMB_DIM, HID_DIM, BATCH_SIZE, vocab, max_trg_len=25).cuda()
	model = Model(vocab, out_len=25, emb_dim=EMB_DIM, hid_dim=HID_DIM).cuda()
	model.eval()

	file = args.model_file
	if os.path.exists(file):
		model.load_state_dict(torch.load(file))
		print('Load model parameters from %s' % file)

	my_test(valid_x, model)


if __name__ == '__main__':
	main()

