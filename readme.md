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
1. we use another thread to preprocess a batch of data, which would not end after the main process end. So you need to press ctrl+c again to terminate the thread.

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

### Performance
* evaluate after 5 epochs training
* use _pyrouge_ to evaluate


Gigaword, with beam search:

|Gigaword||||
|----|--|-|--|
|ROUGE|P|R|F1|
|ROUGE-1|0.640|0.409|0.499|
|ROUGE-2|0.025|0.016|0.019|
|ROUGE-L|0.634|0.405|0.495|

DUC2004, with beam search:

|ROUGE|P|R|F1|
|----|--|-|--|
|ROUGE-1|0.710|0.353|0.472|
|ROUGE-2|0.144|0.071|0.095|
|ROUGE-L|0.699|0.347|0.464|


Gigaword, without beam search:

|ROUGE|P|R|F1|
|----|--|-|--|
|ROUGE-1|0.623|0.449|0.521|
|ROUGE-2|0.029|0.021|0.024|
|ROUGE-L|0.615|0.443|0.515|

DUC2004, without beam search:

|ROUGE|P|R|F1|
|----|--|-|--|
|ROUGE-1|0.703|0.403|0.512|
|ROUGE-2|0.137|0.078|0.099|
|ROUGE-L|0.695|0.398|0.506|

The Rouge-2 score of Gigaword is far less then reported, while the Rouge-1 and Rouge-L score of Gigaword are much more then reported. I don't know why.
