import pandas as pd
import numpy as np
import json, os, time, jieba


def make_words(data_path, stop_words, save_path):
    df = pd.read_csv(data_path)
    stop_words = {el.strip() for el in open(stop_words, 'r', encoding='utf8').readlines() if el.strip()}

    df = [df['Question'], df['Dialogue'], df['Report']]
    df = [el.to_list() for el in df]
    df = sum(df, [])
    df = [el for el in df if type(el) == str]
    df = [jieba.lcut(el) for el in df]

    words = set()
    for sentence in df:
        words.update(sentence)
    words = words - stop_words

    words = pd.DataFrame([[word, i] for i, word in enumerate(words)])
    words.columns = ['index', 'word']

    words.to_csv(save_path, index=False)
