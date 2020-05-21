import tensorflow as tf
import numpy as np
from pyltp import SentenceSplitter,NamedEntityRecognizer,Postagger,Parser,Segmentor
import pandas as pd
from gensim.models.word2vec import LineSentence, Word2Vec
import time, shutil, os
print(tf.__version__)
import matplotlib.pyplot as plt
# shutil.rmtree('checkpoint')

vocabulary_dimension = 100
unites = 128
batch_size = 32

df = pd.read_csv('./AutoMaster_TrainSet.csv')

cws_model = "./ltp_data_v3.4.0/cws.model"

def get_word_list(sentence=None,sentences=None,model=cws_model):
    #得到分词
    segmentor = Segmentor()
    segmentor.load(model)
    if sentences is not None:
        for i, s in enumerate(sentences):
            sentences[i] = list(segmentor.segment(s))
        return sentences
    else:
        word_list = list(segmentor.segment(sentence))
    segmentor.release()
    return word_list



























