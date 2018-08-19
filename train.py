import os
import json
import utils
import torch
import argparse
from layers import Seq2SeqAttention
import logging

parser = argparse.ArgumentParser(description='Selective Encoding for Abstractive Sentence Summarization in DyNet')

parser.add_argument('--gpu', type=int, default='-1', help='GPU ID to use. For cpu, set -1 [default: -1]')
parser.add_argument('--n_epochs', type=int, default=5, help='Number of epochs [default: 3]')
parser.add_argument('--n_train', type=int, default=100000,
					help='Number of training data (up to 3803957 in gigaword) [default: 3803957]')
parser.add_argument('--n_valid', type=int, default=189651,
					help='Number of validation data (up to 189651 in gigaword) [default: 189651])')
parser.add_argument('--batch_size', type=int, default=64, help='Mini batch size [default: 32]')
parser.add_argument('--emb_dim', type=int, default=300, help='Embedding size [default: 256]')
parser.add_argument('--hid_dim', type=int, default=512, help='Hidden state size [default: 256]')
parser.add_argument('--maxout_dim', type=int, default=2, help='Maxout size [default: 2]')
parser.add_argument('--model_file', type=str, default='./models/params_0.pkl')
args = parser.parse_args()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='log/train.log',
    filemode='w'
)

# define a new Handler to log to console as well
console = logging.StreamHandler()
# optional, set the logging level
console.setLevel(logging.INFO)
# set a format which is the same for console use
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
# tell the handler to use this format
console.setFormatter(formatter)
# add the handler to the root logger
logging.getLogger('').addHandler(console)


model_dir = './models'
if not os.path.exists(model_dir):
	os.mkdir(model_dir)

device = torch.device(("cuda:%d" % args.gpu) if args.gpu != -1 else "cpu")
print('using device', device)

def calc_loss(logits, batchY, model):
	loss = model.loss_function(logits.transpose(1,2), batchY)
	return loss

def validate(validX, validY, model):
	with torch.no_grad():
		for _, (batchX, batchY) in enumerate(zip(validX, validY)):
			if args.gpu != -1:
				batchX = torch.tensor(batchX, dtype = torch.long, device=device)
				batchY = torch.tensor(batchY, dtype = torch.long, device=device)
			logits = model(batchX, batchY)
			loss = calc_loss(logits, batchY[:,1:], model) # exclude start token
			return loss

def train(trainX, trainY, validX, validY, model, optimizer, scheduler, epochs=1):
	logging.info("Start to train...")
	for epoch in range(epochs):
		for idx, (batchX, batchY) in enumerate(zip(trainX, trainY)):
			optimizer.zero_grad()

			if args.gpu != -1:
				batchX = torch.tensor(batchX, dtype = torch.long, device=device)
				batchY = torch.tensor(batchY, dtype = torch.long, device=device)
			logits = model(batchX, batchY)
			loss = calc_loss(logits, batchY[:,1:], model) # exclude start token
			loss.backward()
			torch.nn.utils.clip_grad_value_(model.parameters(), 20)

			optimizer.step()
			# scheduler.step()

			if (idx + 1) % 10 == 0:
				train_loss = loss.cpu().detach().numpy()
				valid_loss = validate(validX, validY, model)
				# print('step %d, training loss = %f, validation loss = %f' % (steps, train_loss, valid_loss))
				logging.info('epoch %d, step %d, training loss = %f, validation loss = %f'
							 % (epoch, idx + 1, train_loss, valid_loss))

		# torch.save(model, 'model_%d.pkl' % epoch)
		model.cpu()
		torch.save(model.state_dict(), os.path.join(model_dir, 'params_%d.pkl' % epoch))
		# print('Model saved in dir %s' % model_dir)
		logging.info('Model saved in dir %s' % model_dir)


def decode():
	pass


def main():
	print(args)

	N_EPOCHS = args.n_epochs
	N_TRAIN = args.n_train
	N_VALID = args.n_valid
	BATCH_SIZE = args.batch_size
	EMB_DIM = args.emb_dim
	HID_DIM = args.hid_dim
	MAXOUT_DIM = args.maxout_dim

	data_dir = 'sumdata/'
	TRAIN_X = 'sumdata/train/train.article.txt'
	TRAIN_Y = 'sumdata/train/train.title.txt'
	VALID_X = 'sumdata/train/valid.article.filter.txt'
	VALID_Y = 'sumdata/train/valid.title.filter.txt'

	vocab_file = os.path.join(data_dir, "vocab.json")
	if not os.path.exists(vocab_file):
		utils.build_vocab([TRAIN_X, TRAIN_Y], vocab_file)
	vocab = json.load(open(vocab_file))

	trainX = utils.getDataLoader(TRAIN_X, vocab, n_data=N_TRAIN, batch_size=BATCH_SIZE)
	trainY = utils.getDataLoader(TRAIN_Y, vocab, n_data=N_TRAIN, batch_size=BATCH_SIZE)
	validX = utils.getDataLoader(VALID_X, vocab, n_data=N_VALID, batch_size=BATCH_SIZE)
	validY = utils.getDataLoader(VALID_Y, vocab, n_data=N_VALID, batch_size=BATCH_SIZE)

	model = Seq2SeqAttention(len(vocab), EMB_DIM, HID_DIM, BATCH_SIZE, vocab, device, max_trg_len=25, dropout=0.5)
	if args.gpu != -1:
		model = model.cuda(device)

	model_file = args.model_file
	if os.path.exists(model_file):
		model.load_state_dict(torch.load(model_file))
		logging.info('Load model parameters from %s' % model_file)
		# print('Load model parameters from %s' % model_file)

	optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20000, gamma=0.3)

	train(trainX, trainY, validX, validY, model, optimizer, scheduler, N_EPOCHS)


if __name__ == '__main__':
	main()

