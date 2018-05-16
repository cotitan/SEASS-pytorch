from layers import Seq2SeqAttention
from torch.utils.data import DataLoader
import os
import utils
import torch
import argparse

parser = argparse.ArgumentParser(description='Selective Encoding for Abstractive Sentence Summarization in DyNet')

parser.add_argument('--gpu', type=int, default='1', help='GPU ID to use. For cpu, set -1 [default: -1]')
parser.add_argument('--n_epochs', type=int, default=3, help='Number of epochs [default: 3]')
parser.add_argument('--n_train', type=int, default=20000,
					help='Number of training data (up to 3803957 in gigaword) [default: 3803957]')
parser.add_argument('--n_valid', type=int, default=189651,
					help='Number of validation data (up to 189651 in gigaword) [default: 189651])')
parser.add_argument('--batch_size', type=int, default=32, help='Mini batch size [default: 32]')
parser.add_argument('--vocab_size', type=int, default=124404, help='Vocabulary size [default: 124404]')
parser.add_argument('--emb_dim', type=int, default=200, help='Embedding size [default: 256]')
parser.add_argument('--hid_dim', type=int, default=256, help='Hidden state size [default: 256]')
parser.add_argument('--maxout_dim', type=int, default=2, help='Maxout size [default: 2]')
parser.add_argument('--alloc_mem', type=int, default=10000, help='Amount of memory to allocate [mb] [default: 10000]')
args = parser.parse_args()

model_dir = './models'
if not os.path.exists(model_dir):
	os.mkdir(model_dir)

device = torch.device(("cuda:%d" % args.gpu) if args.gpu != -1 else "cpu")
print('using device', device)


def train(trainX_loader, trainY_loader, model, optimizer, scheduler, epochs=1):
	for epoch in range(epochs):
		for _, (batchX, batchY) in enumerate(zip(trainX_loader, trainY_loader)):

			batchX = torch.tensor(batchX, dtype=torch.long, device=device)
			batchY = torch.tensor(batchY, dtype=torch.long, device=device)
			# batchY = torch.LongTensor(batchY).to(device)
			loss = model(batchX, batchY)
			print(loss)

			loss.backward(retain_graph=True)
			optimizer.step()
			scheduler.step()

		# torch.save(model, 'model_%d.pkl' % epoch)
		torch.save(model.state_dict(), os.path.join(model_dir, 'params_%d.pkl' % epoch))
		print('Model saved in dir %s' % model_dir)


def main():
	print(args)

	N_EPOCHS = args.n_epochs
	N_TRAIN = args.n_train
	N_VALID = args.n_valid
	BATCH_SIZE = args.batch_size
	EMB_DIM = args.emb_dim
	HID_DIM = args.hid_dim
	MAXOUT_DIM = args.maxout_dim
	ALLOC_MEM = args.alloc_mem

	VALID_X = 'PART_III.article'
	VALID_Y = 'PART_III.summary'

	trainX = utils.MyDatasets('PART_I.article', max_len=100, n_data=N_TRAIN)
	trainY = utils.MyDatasets('PART_I.summary', max_len=25, n_data=N_TRAIN)

	print(trainX.datas.shape, trainY.datas.shape)

	trainX_loader = DataLoader(trainX, batch_size=BATCH_SIZE, num_workers=2)
	trainY_loader = DataLoader(trainY, batch_size=BATCH_SIZE, num_workers=2)

	model = Seq2SeqAttention(trainX.vocab_size, EMB_DIM, HID_DIM, BATCH_SIZE, device, max_trg_len=25).cuda(device)

	if os.path.exists(model_dir) and len(os.listdir(model_dir)) > 0:
		file = os.path.join(model_dir, os.listdir(model_dir)[-1])
		model.load_state_dict(torch.load(file))
		print('Load model parameters from %s' % file)

	optimizer = torch.optim.SGD(model.parameters(), lr=3e-5)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.3)

	train(trainX_loader, trainY_loader, model, optimizer, scheduler)


if __name__ == '__main__':
	main()

