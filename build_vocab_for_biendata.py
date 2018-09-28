import utils

dirs = "/home/tiankeke/workspace/SEASS-dynet/data/"
files = ["train_content.txt", "train_title.txt", "valid_content.txt", "valid_title.txt"]
files = [dirs + f for f in files]
freq = utils.build_vocab(files, "sumdata/vocab_biendata.json", min_count=15)
