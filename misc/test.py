import random
import torch
import numpy as np

if __name__ == '__main__':
    checkpoint = torch.load('../data/bert_base_chinese/pytorch_model.bin', map_location=torch.device('cpu'))
    loaded_paras = checkpoint['model_state_dict']
    torch.save(loaded_paras, 'test.bin')
