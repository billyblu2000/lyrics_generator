import random

import pandas as pd
import torch
import numpy as np
from transformers import BertTokenizer

from models.bert_nsp_pretrain_utils.bert_nsp_pretrain_config import ModelConfig

if __name__ == '__main__':
    data = pd.read_csv('../data/phrase_database_d0.csv')
    max_sentence_count = {}
    print(len(data['phrase']))
    for i in range(len(data['phrase'])):
        pos = int(data['sentence_position_in_song'][i])
        ind = int(data['rhy_index'][i])
        # print(pos, ind)
        if pos == 19 and ind == 1:
            print(i)
        if data['rhy_index'][i] not in max_sentence_count:
            max_sentence_count[ind] = pos
        else:
            if max_sentence_count[ind] < pos:
                max_sentence_count[ind] = pos
    print(max_sentence_count)
    rhy = pd.read_csv('../data/rhythmic_index.csv', index_col=0)
    max_sentence_count = [max_sentence_count[i] for i in range(len(max_sentence_count))]
    rhy['num_sentence_count'] = max_sentence_count
    rhy.to_csv('rhythmic_index_new.csv', index=True)