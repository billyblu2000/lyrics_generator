import os

import torch
from torch.utils.tensorboard import SummaryWriter
from transformers import BertConfig


class ModelConfig:

    def __init__(self):
        self.project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # ========== wike2 数据集相关配置
        # self.dataset_dir = os.path.join(self.project_dir, 'data', 'WikiText')
        # self.pretrained_model_dir = os.path.join(self.project_dir, "bert_base_uncased_english")
        # self.train_file_path = os.path.join(self.dataset_dir, 'wiki.train.tokens')
        # self.val_file_path = os.path.join(self.dataset_dir, 'wiki.valid.tokens')
        # self.test_file_path = os.path.join(self.dataset_dir, 'wiki.test.tokens')
        # self.data_name = 'wiki2'

        # ========== songci 数据集相关配置
        self.dataset_dir = os.path.join(self.project_dir, 'data', 'SongCi')
        self.pretrained_model_dir = os.path.join(self.project_dir, "bert_base_chinese")
        self.train_file_path = os.path.join(self.dataset_dir, 'songci.train.txt')
        self.val_file_path = os.path.join(self.dataset_dir, 'songci.valid.txt')
        self.test_file_path = os.path.join(self.dataset_dir, 'songci.test.txt')
        self.data_name = 'songci'

        # 如果需要切换数据集，只需要更改上面的配置即可
        self.vocab_path = os.path.join(self.pretrained_model_dir, 'vocab.txt')
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model_save_dir = os.path.join(self.project_dir, 'cache')
        self.logs_save_dir = os.path.join(self.project_dir, 'logs')
        self.model_save_path = os.path.join(self.model_save_dir, f'model_{self.data_name}.bin')
        self.writer = SummaryWriter(f"runs/{self.data_name}")
        self.is_sample_shuffle = True
        self.use_embedding_weight = True
        self.batch_size = 16
        self.max_sen_len = None  # 为None时则采用每个batch中最长的样本对该batch中的样本进行padding
        self.pad_index = 0
        self.random_state = 2022
        self.learning_rate = 4e-5
        self.weight_decay = 0.1
        self.masked_rate = 0.15
        self.masked_token_rate = 0.8
        self.masked_token_unchanged_rate = 0.5
        self.use_torch_multi_head = False  # False表示使用model/BasicBert/MyTransformer中的多头实现
        self.epochs = 200
        self.model_val_per_epoch = 1
        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)
        bert_config_path = os.path.join(self.pretrained_model_dir, "config.json")
        bert_config = BertConfig.from_json_file(bert_config_path)
        for key, value in bert_config.__dict__.items():
            self.__dict__[key] = value
