from layers import SelectiveBiGRU, AttentionDecoder
from torch.utils.data import DataLoader
import utils
import torch
import os
import time
import argparse

parser = argparse.ArgumentParser(description='Selective Encoding for Abstractive Sentence Summarization in DyNet')

parser.add_argument('--gpu', type=str, default='0', help='GPU ID to use. For cpu, set -1 [default: -1]')
parser.add_argument('--n_epochs', type=int, default=3, help='Number of epochs [default: 3]')
parser.add_argument('--n_train', type=int, default=3803957, help='Number of training data (up to 3803957 in gigaword) [default: 3803957]')
parser.add_argument('--n_valid', type=int, default=189651, help='Number of validation data (up to 189651 in gigaword) [default: 189651])')
parser.add_argument('--batch_size', type=int, default=8, help='Mini batch size [default: 32]')
parser.add_argument('--vocab_size', type=int, default=124404, help='Vocabulary size [default: 124404]')
parser.add_argument('--emb_dim', type=int, default=200, help='Embedding size [default: 256]')
parser.add_argument('--hid_dim', type=int, default=256, help='Hidden state size [default: 256]')
parser.add_argument('--maxout_dim', type=int, default=2, help='Maxout size [default: 2]')
parser.add_argument('--alloc_mem', type=int, default=10000, help='Amount of memory to allocate [mb] [default: 10000]')
args = parser.parse_args()


def train(trainX_loader, trainY_loader, encoder, decoder, enc_optimizer, dec_optimizer, epochs=10):
    for epoch in range(epochs):
        for batchX, batchY in enumerate(zip(trainX_loader, trainY_loader)):
            enc_optimizer.zero_grad()
            dec_optimizer.zero_grad()

            states, hidden = encoder(batchX)
            loss = decoder(batchY, hidden, states)
            print(loss)

            loss.backward(retain_graph=True)
            enc_optimizer.step()
            dec_optimizer.step()

        
def main():
    
    print(args)

    N_EPOCHS   = args.n_epochs
    N_TRAIN    = args.n_train
    N_VALID    = args.n_valid
    BATCH_SIZE = args.batch_size
    EMB_DIM    = args.emb_dim
    HID_DIM    = args.hid_dim
    MAXOUT_DIM = args.maxout_dim
    ALLOC_MEM  = args.alloc_mem

    VALID_X = 'PART_III.article'
    VALID_Y = 'PART_III.summary'

    trainX = utils.MyDatasets('PART_III.article', max_len=100, n_data=N_TRAIN)
    trainY = utils.MyDatasets('PART_III.summary', max_len=25, n_data=N_TRAIN)

    trainX_loader = DataLoader(trainX, batch_size=BATCH_SIZE, num_workers=4)
    trainY_loader = DataLoader(trainY, batch_size=BATCH_SIZE, num_workers=4)

    encoder = SelectiveBiGRU(trainX.vocab_size, EMB_DIM, HID_DIM//2)
    decoder = AttentionDecoder(trainX.vocab_size, EMB_DIM, HID_DIM)
    
    enc_optimizer = torch.optim.SGD(encoder.parameters(), lr=0.01, weight_decay=0.9)
    dec_optimizer = torch.optim.SGD(decoder.parameters(), lr=0.01, weight_decay=0.9)

    train(trainX_loader, trainY_loader, encoder, decoder, enc_optimizer, dec_optimizer)

if __name__ == '__main__':
    main()

