import os
import json
import torch
import argparse
import numpy as np
from utils import BatchManager, load_data
from Model import Model
from Beam import Beam
import torch.nn.functional as F
import utils

parser = argparse.ArgumentParser(description='Selective Encoding for Abstractive Sentence Summarization in pytorch')

parser.add_argument('--n_test', type=int, default=1961, help='Number of test data (up to 1951 in gigaword)')
parser.add_argument('--input_file', type=str, default="sumdata/Giga/input.txt", help='input file')
parser.add_argument('--output_dir', type=str, default="sumdata/Giga/systems/", help='')
parser.add_argument('--batch_size', type=int, default=64, help='Mini batch size [default: 32]')
parser.add_argument('--ckpt_file', type=str, default='./ckpts/params_0.pkl', help='model file path')
parser.add_argument('--search', type=str, default='greedy', help='greedy/beam')
parser.add_argument('--beam_width', type=int, default=12, help='beam search width')
args = parser.parse_args()
print(args)

if not os.path.exists(args.ckpt_file):
	raise FileNotFoundError("model file not found")


def print_summaries(summaries, vocab):
	"""
	param summaries: in shape (seq_len, batch)
	"""
	i2w = {key: value for value, key in vocab.items()}

	for idx in range(len(summaries)):
		fout = open(os.path.join(args.output_dir, "%d.txt" % idx), "w")
		line = [i2w[tok] for tok in summaries[idx] if i2w[tok] not in ["</s>", "<pad>"]]
		fout.write(" ".join(line) + "\n")
		fout.close()


def greedy(model, batch_x, max_trg_len=15):
	enc_outs, hidden = model.encode(batch_x)
	hidden = model.init_decoder_hidden(hidden)
	mask = batch_x.eq(model.vocab['<pad>']).unsqueeze(1).cuda()
	
	words = []
	word = torch.ones(hidden.shape[1], dtype=torch.long).cuda() * model.vocab["<s>"]
	for _ in range(max_trg_len):
		logit, hidden = model.decode(word, enc_outs, hidden, mask)
		word = torch.argmax(logit, dim=-1)
		words.append(word.cpu().numpy())
	return np.array(words).T


def beam_search(model, batch_x, max_trg_len=15, k=args.beam_width):
	enc_outs, hidden = model.encode(batch_x)
	hidden = model.init_decoder_hidden(hidden)

	beams = [Beam(k, model.vocab, hidden[:,i,:])
			for i in range(batch_x.shape[0])]
	
	for _ in range(max_trg_len):
		for j in range(len(beams)):
			hidden = beams[j].get_hidden_state()
			word = beams[j].get_current_word()
			enc_outs_j = enc_outs[j].unsqueeze(0).expand(k, -1, -1)
			logit, hidden = model.decode(word, enc_outs_j, hidden)
			# logit: [k x V], hidden: [k x hid_dim]
			log_probs = torch.log(F.softmax(logit, -1))
			beams[j].advance_(log_probs, hidden)

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


def my_test(test_x, model):
	summaries = []
	with torch.no_grad():
		for _ in range(test_x.steps):
			batch_x = test_x.next_batch().cuda()
			if args.search == "greedy":
				summary = greedy(model, batch_x)
			elif args.search == "beam":
				summary = beam_search(model, batch_x)
			else:
				raise NameError("Unknown search method")
			summaries.extend(summary)
	print_summaries(summaries, model.vocab)
	print("Done!")


def main():

	N_TEST = args.n_test
	BATCH_SIZE = args.batch_size

	# vocab = json.load(open('sumdata/vocab.json'))

	embedding_path = '/home/kaiying/coco/embeddings/giga-256d.bin'
	vocab, embeddings = utils.load_word2vec_embedding(embedding_path)

	test_x = BatchManager(load_data(args.input_file, vocab, N_TEST), BATCH_SIZE)
	# model = Seq2SeqAttention(len(vocab), EMB_DIM, HID_DIM, BATCH_SIZE, vocab, max_trg_len=25).cuda()
	model = Model(vocab, out_len=15, emb_dim=256, hid_dim=512, embeddings=embeddings).cuda()
	model.eval()

	file = args.ckpt_file
	if os.path.exists(file):
		saved_state = torch.load(file)
		model.load_state_dict(saved_state['state_dict'])
		print('Load model parameters from %s' % file)

		my_test(test_x, model)


if __name__ == '__main__':
	main()

