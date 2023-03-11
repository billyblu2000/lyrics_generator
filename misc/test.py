import random

import pandas as pd
import torch
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from transformers import BertTokenizer

from models.bert_nsp_pretrain_utils.bert_nsp_pretrain_config import ModelConfig

if __name__ == '__main__':
    # size = 2
    # x = np.arange(size)
    # a = [3.35, 2.7]
    #
    # total_width, n = 2, 3
    # width = total_width / n
    # x = x - (total_width - width) / 2
    # plt.figure(dpi=1000)
    # plt.ylim(2.0, 4.0)
    # plt.ylabel('Ratings')
    # plt.bar(x + width, a, width=width, label='Human', color='#A2C0C2')
    # plt.xticks([0, 1], labels=['Humam', 'Ours'])
    # plt.show()


    size = 3
    x = np.arange(size)
    a = [3.5, 3.21, 3.51]

    total_width, n = 2, 3
    width = total_width / n
    x = x - (total_width - width) / 2
    plt.figure(dpi=1000)
    plt.ylim(2.8, 3.8)
    plt.ylabel('Ratings')
    plt.bar(x + width, a, width=width, label='Human', color='#A2C0C2')
    plt.xticks([0, 1, 2], labels=['Humam', 'Baseline', 'Ours'])
    plt.show()

    size = 3
    x = np.arange(size)
    a = [3.829268293, 3.902439024, 3.804878049]
    b = [3.195121951, 2.902439024, 3.219512195]
    c = [3.434782609, 3.608695652, 3.52173913]

    total_width, n = 0.8, 3
    width = total_width / n
    x = x - (total_width - width) / 2
    plt.figure(dpi=1000)
    plt.ylim(2.0, 4.5)
    plt.ylabel('Ratings')
    plt.bar(x, a, width=width, label='Human', color='#A2C0C2')
    plt.bar(x + width, b, width=width, label='Baseline', color='#4C4C6A')
    plt.bar(x + 2 * width, c, width=width, label='Ours', color='#90A7AF')
    plt.xticks([0, 1, 2], labels=['Fluency', 'Consistency', 'Artistry'])
    plt.legend()
    plt.show()