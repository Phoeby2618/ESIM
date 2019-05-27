# ESIM
A tensorflow implementation of ESIM model ，

[Enhanced LSTM for Natural Language Inference](<https://arxiv.org/pdf/1609.06038.pdf>)

which is on the strength of decompAtt.[A Decomposable Attention Model for Natural Language Inference](<https://aclweb.org/anthology/D16-1244>)	

## Requirements

- Python>=3
- TensorFlow>=1.12
- Numpy
- Gensim
- NLTK >=3.4

## Dataset

For Textual Entailment dataset,

such as [The Stanford Natural Language Inference (SNLI) ](<https://nlp.stanford.edu/projects/snli/>),[The Scitail dataset](<http://data.allenai.org/scitail/>) etc.

## Usage

1. Download dataset

2. Modify file paths in ` config.py` 

3. Training model

   ```
   python run.py --train
   ```

   

