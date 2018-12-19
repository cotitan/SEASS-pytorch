import os
import json
import torch
import argparse
import numpy as np
from utils import BatchManager, load_data
from layers import Seq2SeqAttention
from Model import Model

parser = argparse.ArgumentParser(description='Selective Encoding for Abstractive Sentence Summarization in pytorch')

parser.add_argument('--n_valid', type=int, default=189651,
					help='Number of validation data (up to 189651 in gigaword) [default: 189651])')
parser.add_argument('--input_file', type=str, default="sumdata/Giga/input.txt", help='input file')
parser.add_argument('--batch_size', type=int, default=64, help='Mini batch size [default: 32]')
parser.add_argument('--emb_dim', type=int, default=300, help='Embedding size [default: 256]')
parser.add_argument('--hid_dim', type=int, default=512, help='Hidden state size [default: 256]')
parser.add_argument('--maxout_dim', type=int, default=2, help='Maxout size [default: 2]')
parser.add_argument('--model_file', type=str, default='./models/params_0.pkl', help='model file path')
args = parser.parse_args()
print(args)

if not os.path.exists(args.model_file):
	raise FileNotFoundError("model file not found")

start_tok = "<s>"
end_tok = "</s>"
unk_tok = "UNK"
pad_tok = "<pad>"


def print_summaries(summaries, vocab):
	"""
	param summaries: list of shape (seq_len, batch_tensor)
	"""
	i2w = {key: value for value, key in vocab.items()}
	summaries = np.array(summaries).T

	for stnc in summaries:
		line = [i2w[tok] for tok in stnc if tok != vocab["</s>"]]
		print(" ".join(line))


def my_test(valid_x, model):
	with torch.no_grad():
		for _ in range(valid_x.steps):
			batch_x = valid_x.next_batch().cuda()
			summaries = model(batch_x, None, test=True)
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

	file = args.model_file
	if os.path.exists(file):
		model.load_state_dict(torch.load(file))
		print('Load model parameters from %s' % file)

	my_test(valid_x, model)


if __name__ == '__main__':
	main()

