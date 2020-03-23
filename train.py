import os, sys
import re
import string
import pathlib
import random
from collections import Counter, OrderedDict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import spacy
from tqdm import tqdm


import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import torchtext
from torchtext import data, vocab
from torchtext.vocab import Vectors, GloVe


from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


from utils import BatchGenerator, fit
from utils import SimpleGRU, SimpleLSTM, ConcatPoolingGRUAdaptive, StackedGRU


import warnings
warnings.filterwarnings('ignore')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


df = pd.read_csv("Statin_Data.csv")
df = df[df.columns[:3]]
print("Read Data file")

embedding_dim = 100
gloveType = '6B'
n_hidden = 128
n_out = 8
train_epochs = 10
dropout = 0.8


nlp = spacy.load('en',disable=['parser', 'tagger', 'ner'])
def tokenizer(s): return [w.text.lower() for w in nlp(tweet_clean(s))]


def tweet_clean(text):
    text = re.sub(r'[^A-Za-z0-9]+', ' ', text) # remove non alphanumeric character
    text = re.sub(r'https?:/\/\S+', ' ', text) # remove links
    return text.strip()


txt_field = data.Field(sequential=True, tokenize=tokenizer, include_lengths=True, use_vocab=True)
label_field = data.Field(sequential=False, use_vocab=True, pad_token=None, unk_token=None)

train_val_fields = [
    ('tweet_id', None),
    ('text', txt_field),
    ('Category',label_field)
]


### Creating Train-Valid split

X = df.loc[:,df.columns!='Category']
y = df.Category

trainx, testx, trainy, testy = train_test_split(X,y, test_size = 0.2, random_state= 123)

trainDf = pd.concat([trainx,trainy], axis = 1)
testDf = pd.concat([testx,testy], axis = 1)

trainDf.to_csv("train.csv",index = False, index_label = False)
testDf.to_csv("test.csv",index = False, index_label = False)


trainds,validds = data.TabularDataset.splits(path='./', format='csv', train='train.csv',
                                             validation = 'test.csv',
                                             fields=train_val_fields,
                                            skip_header=True)



### DEFINING THE EMBEDDINGS
vec = GloVe(name=gloveType, cache = './Glove/', dim=embedding_dim)
print("Read Embedding File")


## Building the vocab
txt_field.build_vocab(trainds, validds, max_size=100000, vectors=vec)

label_field.build_vocab(trainds)
traindl = data.BucketIterator(dataset=(trainds), 
                                            batch_size=4, 
                                            sort_key=lambda x: len(x.text), 
                                            device=device, 
                                            sort_within_batch=True, 
                                            repeat=False)


validdl = data.BucketIterator(dataset=(validds), 
                                            batch_size=6, 
                                            sort_key=lambda x: len(x.text), 
                                            device=device, 
                                            sort_within_batch=True, 
                                            repeat=False)



train_batch_it = BatchGenerator(traindl, 'text', 'Category')
valid_batch_it = BatchGenerator(validdl, 'text', 'Category')


vocab_size = len(txt_field.vocab)




model = SimpleGRU(vocab_size, embedding_dim, n_hidden, n_out, trainds.fields['text'].vocab.vectors, dropout = dropout).to(device)

opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 1e-3)


fit(model=model, train_dl=train_batch_it, val_dl=valid_batch_it, loss_fn=F.nll_loss, opt=opt, epochs=train_epochs)






