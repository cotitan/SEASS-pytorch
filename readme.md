### Pytorch implementation for paper "Selective Encoding for Abstractive Sentence Summarization" (Haven't finish yet)
- author: Kirk
- mail: cotitan@outlook.com

### Requirments
- pytorch==0.4.0
- numpy==1.12.1+
- python=3.5+

### Data
Training and evaluation data for Gigaword is available https://drive.google.com/open?id=0B6N7tANPyVeBNmlSX19Ld2xDU1E

Training and evaluation data for CNN/DM is available https://s3.amazonaws.com/opennmt-models/Summary/cnndm.tar.gz

### Noticement
1. we use another thread to preprocess a batch of data, which would not terminate after the main process terminate. So you need to press ctrl+c again to terminate the thread.

### Directories:
```
.
├── Beam.py
├── Model.py
├── mytest.py
├── train.py
├── utils.py
├── sumdata/
|   ├── DUC2003/
|   ├── DUC2004/
|   ├── Giga/
|   ├── train/
|   └── vocab.json # will be built automatically if not exists
├── readme.md
├── log/
└── models/
```
Make sure your project contains the folders above.

### How-to
1. Run _python train.py_ to train, it takes about 3.5h per epoch.
2. Run _python mytest.py_ to generate summaries

### TODO
1. learning rate decay, which is essential
