from layers import SelectiveBiGRU, AttentionDecoder
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
parser.add_argument('--batch_size', type=int, default=32, help='Mini batch size [default: 32]')
parser.add_argument('--vocab_size', type=int, default=124404, help='Vocabulary size [default: 124404]')
parser.add_argument('--emb_dim', type=int, default=256, help='Embedding size [default: 256]')
parser.add_argument('--hid_dim', type=int, default=256, help='Hidden state size [default: 256]')
parser.add_argument('--maxout_dim', type=int, default=2, help='Maxout size [default: 2]')
parser.add_argument('--alloc_mem', type=int, default=10000, help='Amount of memory to allocate [mb] [default: 10000]')
args = parser.parse_args()


def train(articles, summaries, encoder, decoder, enc_optimizer, dec_optimizer, epochs=50):
    for epoch in range(epochs):
        for i in range(articles.shape[0]):
            enc_optimizer.zero_grad()
            dec_optimizer.zero_grad()

            states, hidden = encoder(articles[i])
            loss = decoder(summaries[i], hidden, states)
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

    train_datas = utils.MyDatasets('PART_III.article', 'PART_III.summary', n_data=N_TRAIN)
    valid_datas = utils.MyDatasets(VALID_X, VALID_Y)

    encoder = SelectiveBiGRU(train_datas.vocab_size, EMB_DIM, HID_DIM//2)
    decoder = AttentionDecoder(train_datas.vocab_size, EMB_DIM, HID_DIM)
    print(train_datas.__len__())
    
    enc_optimizer = torch.optim.SGD(encoder.parameters(), lr=0.01, weight_decay=0.9)
    dec_optimizer = torch.optim.SGD(decoder.parameters(), lr=0.01, weight_decay=0.9)

    x, y = train_datas[0]
    train(x.view(1,-1), y.view(1,-1), encoder, decoder, enc_optimizer, dec_optimizer)

if __name__ == '__main__':
    main()

