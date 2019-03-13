import os
import json
import utils
import torch
import argparse
import shutil
from Model import Model
from utils import BatchManager, load_data
from tensorboardX import SummaryWriter
import logging

parser = argparse.ArgumentParser(description='Selective Encoding for Abstractive Sentence Summarization in DyNet')

parser.add_argument('--n_epochs', type=int, default=10, help='Number of epochs [default: 3]')
parser.add_argument('--n_train', type=int, default=3803900,
					help='Number of training data (up to 3803957 in gigaword) [default: 3803957]')
parser.add_argument('--n_valid', type=int, default=189651,
					help='Number of validation data (up to 189651 in gigaword) [default: 189651])')
parser.add_argument('--batch_size', type=int, default=64, help='Mini batch size [default: 32]')
parser.add_argument('--ckpt_file', type=str, default='./ckpts/params_0.pkl')
parser.add_argument('--data_dir', type=str, default='/home/kaiying/coco/datas/sumdata/')
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


model_dir = './ckpts'
if not os.path.exists(model_dir):
	os.mkdir(model_dir)


def run_batch(valid_x, valid_y, model):
	batch_x = valid_x.next_batch().cuda()
	batch_y = valid_y.next_batch().cuda()
	mask = batch_x.eq(model.vocab['<pad>']).unsqueeze(1).cuda()

	outputs, hidden = model.encode(batch_x)
	hidden = model.init_decoder_hidden(hidden)

	# logits = torch.zeros(batch_y.shape[0], 0, model.n_vocab).cuda()
	# for i in range(batch_y.shape[1]-1):
	# 	logit, hidden = model.decode(batch_y[:, i], outputs, hidden, mask)
	# 	logits = torch.cat([logits, logit.unsqueeze(1)], dim=1)
	# loss = model.loss_layer(logits.view(-1, model.n_vocab),
	# 						batch_y[:, 1:].contiguous().view(-1))
	loss = 0
	for i in range(batch_y.shape[1] - 1):
		logit, hidden = model.decode(batch_y[:, i], outputs, hidden, mask)
		loss += model.loss_layer(logit, batch_y[:, i+1])
	loss /= batch_y.shape[1]
	return loss


def train(train_x, train_y, valid_x, valid_y, model, optimizer, scheduler, epoch=0, epochs=10):
	logging.info("Start to train...")
	n_batches = train_x.steps
	for epoch in range(epoch, epochs):
		if os.path.exists('runs/epoch%d' % epoch):
			shutil.rmtree('runs/epoch%d' % epoch)
		writer = SummaryWriter('runs/epoch%d' % epoch)
		for idx in range(n_batches):
			optimizer.zero_grad()

			loss = run_batch(train_x, train_y, model)
			loss.backward()  # do not use retain_graph=True
			torch.nn.utils.clip_grad_value_(model.parameters(), 5)

			optimizer.step()

			if (idx + 1) % 50 == 0:
				train_loss = loss.cpu().detach().numpy()
				model.eval()
				with torch.no_grad():
					valid_loss = run_batch(valid_x, valid_y, model)
				logging.info('epoch %d, step %d, training loss = %f, validation loss = %f'
							 % (epoch, idx + 1, train_loss, valid_loss))
				model.train()
				writer.add_scalar('train_loss', train_loss, (idx + 1) / 50)
				writer.add_scalar('valid_loss', valid_loss, (idx + 1) / 50)
		if epoch < 6:
			scheduler.step()
		writer.close()
		saved_state = {'epoch': epoch + 1, 'lr': optimizer.param_groups[0]['lr'],
					   'state_dict': model.state_dict()}
		torch.save(saved_state, os.path.join(model_dir, 'params_%d.pkl' % epoch))
		logging.info('Model saved in dir %s' % model_dir)


def main():
	print(args)

	N_EPOCHS = args.n_epochs
	N_TRAIN = args.n_train
	N_VALID = args.n_valid
	BATCH_SIZE = args.batch_size

	data_dir = args.data_dir
	TRAIN_X = os.path.join(data_dir, 'train/train.article.txt')
	TRAIN_Y = os.path.join(data_dir, 'train/train.title.txt')
	VALID_X = os.path.join(data_dir, 'train/valid.article.filter.txt')
	VALID_Y = os.path.join(data_dir, 'train/valid.title.filter.txt')

	"""
	vocab_file = os.path.join(data_dir, "vocab.json")
	if not os.path.exists(vocab_file):
		utils.build_vocab([TRAIN_X, TRAIN_Y], vocab_file, n_vocab=80000)
	vocab = json.load(open(vocab_file))
	"""
		
	embedding_path = '/home/kaiying/coco/embeddings/giga-256d.bin'
	vocab, embeddings = utils.load_word2vec_embedding(embedding_path)
	print(len(vocab), embeddings.shape)

	train_x = BatchManager(load_data(TRAIN_X, vocab, N_TRAIN), BATCH_SIZE)
	train_y = BatchManager(load_data(TRAIN_Y, vocab, N_TRAIN), BATCH_SIZE)

	valid_x = BatchManager(load_data(VALID_X, vocab, N_VALID), BATCH_SIZE)
	valid_y = BatchManager(load_data(VALID_Y, vocab, N_VALID), BATCH_SIZE)

	# model = Seq2SeqAttention(len(vocab), EMB_DIM, HID_DIM, BATCH_SIZE, vocab, max_trg_len=25).cuda()
	model = Model(vocab, out_len=15, emb_dim=256, hid_dim=512, embeddings=embeddings).cuda()
	# model.embedding_look_up.to(torch.device("cpu"))

	ckpt_file = args.ckpt_file
	saved_state = {'lr': 0.001, 'epoch': 0}
	if os.path.exists(ckpt_file):
		saved_state = torch.load(ckpt_file)
		model.load_state_dict(saved_state['state_dict'])
		logging.info('Load model parameters from %s' % ckpt_file)

	optimizer = torch.optim.Adam(model.parameters(), lr=saved_state['lr'])
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
	scheduler.step()

	train(train_x, train_y, valid_x, valid_y, model, optimizer,
		  scheduler, saved_state['epoch'], N_EPOCHS)


if __name__ == '__main__':
	main()

