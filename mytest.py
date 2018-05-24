import os
import json
import utils
import torch
import argparse
from layers import Seq2SeqAttention

parser = argparse.ArgumentParser(description='Selective Encoding for Abstractive Sentence Summarization in DyNet')

parser.add_argument('--gpu', type=int, default='0', help='GPU ID to use. For cpu, set -1 [default: -1]')
parser.add_argument('--n_valid', type=int, default=189651,
					help='Number of validation data (up to 189651 in gigaword) [default: 189651])')
parser.add_argument('--batch_size', type=int, default=64, help='Mini batch size [default: 32]')
parser.add_argument('--emb_dim', type=int, default=200, help='Embedding size [default: 256]')
parser.add_argument('--hid_dim', type=int, default=256, help='Hidden state size [default: 256]')
parser.add_argument('--maxout_dim', type=int, default=2, help='Maxout size [default: 2]')
parser.add_argument('--model_file', type=str, default='./models/params_0.pkl', help='Maxout size [default: 2]')
args = parser.parse_args()
print(args)

model_dir = './models'
if not os.path.exists(model_dir):
	os.mkdir(model_dir)

device = torch.device(("cuda:%d" % args.gpu) if args.gpu != -1 else "cpu")
print('using device', device)


def validate(validX, validY, model):
	with torch.no_grad():
		loss = torch.zeros(1, dtype=torch.float, device=device)
		for idx, (batchX, batchY) in enumerate(zip(validX, validY)):
			batchX = torch.tensor(batchX, dtype=torch.long, device=device)
			batchY = torch.tensor(batchY, dtype=torch.long, device=device)
			loss += model(batchX, batchY)
		loss /= idx
		print('validation loss:', loss)


def mytest(validX, validY, model, st='<s>', ed='</s>'):
	i2w = {key: value for value, key in model.vocab.items()}

	with torch.no_grad():
		for _, (batchX, batchY) in enumerate(zip(validX, validY)):
			batchX = torch.tensor(batchX, dtype=torch.long, device=device)
			batchY = torch.tensor(batchY, dtype=torch.long, device=device)
			summaries = model(batchX, batchY, test=True)

			sums = summaries.cpu().numpy().squeeze()

			for i in range(sums.shape[0]):
				line = ''
				for idx in sums[i][1:]:
					if idx == model.vocab[ed]:
						print(line)
						break
					else:
						line += str(i2w[int(idx)])


def main():

	N_VALID = args.n_valid
	BATCH_SIZE = args.batch_size
	EMB_DIM = args.emb_dim
	HID_DIM = args.hid_dim

	VALID_X = 'PART_III.article'
	VALID_Y = 'PART_III.summary'

	validX = utils.getDataLoader(VALID_X, max_len=100, n_data=N_VALID, batch_size=BATCH_SIZE)
	validY = utils.getDataLoader(VALID_Y, max_len=25, n_data=N_VALID, batch_size=BATCH_SIZE)

	vocab = json.load(open('data/vocab.json'))
	model = Seq2SeqAttention(len(vocab), EMB_DIM, HID_DIM, BATCH_SIZE, vocab, device, max_trg_len=25).cuda(device)

	file = args.model_file
	if os.path.exists(file):
		model.load_state_dict(torch.load(file))
		print('Load model parameters from %s' % file)

	mytest(validX, validY, model)


if __name__ == '__main__':
	main()

