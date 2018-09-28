### Pytorch implementation for paper "Selective Encoding for Abstractive Sentence Summarization" (Haven't finish yet)
- author: Kirk
- mail: cotitan@outlook.com

### Requirments
- pytorch==0.4.0
- numpy==1.12.1+
- python=3.5+

### Data
Training and evaluation data for Gigaword is available https://drive.google.com/open?id=0B6N7tANPyVeBNmlSX19Ld2xDU1E

Training and evaluation for CNN/DM is available https://s3.amazonaws.com/opennmt-models/Summary/cnndm.tar.gz

### Noticement
1. ~~When running train.py, the console may print out EOFError or ConnectionResetError, which is caused by DataLoader module of pytorch. I don't know why but it does not influence the training process. Neglect It!~~ Set num_workers=0 (argument of DataLoader) to avoid these error messages.
2. This project haven't finish yet, there are bugs to be fix, and modules to be implement.

### How-to
1. Run _python train.py_ to train
2. Run _python mytest.py_ to generate summaries


### TODO
1. ~~Implement maxout layer~~
2. ROUGE metric
3. Attention calculation
4. ~~Fix bugs on decoder~~ 
