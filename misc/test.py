import random
import torch
import numpy as np
from transformers import BertTokenizer

from models.bert_nsp_pretrain_utils.bert_nsp_pretrain_config import ModelConfig

if __name__ == '__main__':
    a = torch.load('../data/phrase_database_d0__mlNone_rs2022_mr15_mtr8_mtur5_nspnc3_only_mlmFalse.pt')['data'][:100]
    config = ModelConfig()
    tok = BertTokenizer.from_pretrained(config.pretrained_model_dir)
    for i in a:
        print(i[0])
        print(tok.decode(i[0]))
        print(i[3])
