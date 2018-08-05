import os
import json
import utils
import torch
import argparse
from layers import Seq2SeqAttention

parser = argparse.ArgumentParser(description='Selective Encoding for Abstractive Sentence Summarization in DyNet')

parser.add_argument('--gpu', type=int, default='-1', help='GPU ID to use. For cpu, set -1 [default: -1]')
parser.add_argument('--n_valid', type=int, default=189651,
					help='Number of validation data (up to 189651 in gigaword) [default: 189651])')
parser.add_argument('--batch_size', type=int, default=64, help='Mini batch size [default: 32]')
parser.add_argument('--emb_dim', type=int, default=200, help='Embedding size [default: 256]')
parser.add_argument('--hid_dim', type=int, default=256, help='Hidden state size [default: 256]')
parser.add_argument('--maxout_dim', type=int, default=2, help='Maxout size [default: 2]')
parser.add_argument('--model_file', type=str, default='./models/params_0.pkl', help='model file path')
args = parser.parse_args()
print(args)

if not os.path.exists(args.model_file):
	raise FileNotFoundError("model file not found")

device = torch.device(("cuda:%d" % args.gpu) if args.gpu != -1 else "cpu")
print('using device', device)

def printSum(summaries, vocab, st, ed):
	i2w = {key: value for value, key in vocab.items()}
	if isinstance(summaries, list):
		for sum in summaries:
			tokens = [i2w[int(idx.numpy())] for idx in sum[0]]
			print(" ".join(tokens))

	else:
		sums = summaries.cpu().numpy().squeeze()
		for i in range(sums.shape[0]):
			line = ''
			for idx in sums[i]:
				if idx == vocab[ed]:
					print(line)
					break
				else:
					line += str(i2w[int(idx)]) + " "
			print(line)


### test
def mytest(validX, validY, model, st='<s>', ed='</s>'):

	with torch.no_grad():
		for _, (batchX, batchY) in enumerate(zip(validX, validY)):
			if args.gpu != -1:
				batchX = torch.tensor(batchX, dtype = torch.long, device=device)
				batchY = torch.tensor(batchY, dtype = torch.long, device=device)
			summaries = model(batchX, batchY, test=True)
			printSum(summaries, model.vocab, st, ed)

def main():

	N_VALID = args.n_valid
	BATCH_SIZE = args.batch_size
	EMB_DIM = args.emb_dim
	HID_DIM = args.hid_dim

	VALID_X = 'sumdata/Giga/input.txt'
	VALID_Y = 'sumdata/Giga/task1_ref0.txt'

	vocab = json.load(open('sumdata/vocab.json'))
	validX = utils.getDataLoader(VALID_X, vocab, n_data=N_VALID, batch_size=BATCH_SIZE)
	validY = utils.getDataLoader(VALID_Y, vocab, n_data=N_VALID, batch_size=BATCH_SIZE)

	model = Seq2SeqAttention(len(vocab), EMB_DIM, HID_DIM, BATCH_SIZE, vocab, device, max_trg_len=25)
	if args.gpu != -1:
		model = model.cuda(device)

	file = args.model_file
	if os.path.exists(file):
		model.load_state_dict(torch.load(file))
		print('Load model parameters from %s' % file)

	mytest(validX, validY, model)


if __name__ == '__main__':
	main()

