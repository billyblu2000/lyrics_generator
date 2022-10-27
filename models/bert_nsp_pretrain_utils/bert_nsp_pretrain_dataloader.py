import random

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.bert_nsp_pretrain_utils.bert_nsp_pretrain_utils import cache, build_vocab, pad_sequence


class LoadBertPretrainingDataset(object):
    def __init__(self,
                 vocab_path='./vocab.txt',
                 tokenizer=None,
                 batch_size=32,
                 max_sen_len=None,
                 max_position_embeddings=512,
                 pad_index=0,
                 is_sample_shuffle=True,
                 random_state=2021,
                 data_name='ci',
                 masked_rate=0.15,
                 masked_token_rate=0.8,
                 masked_token_unchanged_rate=0.5):
        self.tokenizer = tokenizer
        self.vocab = build_vocab(vocab_path)
        self.PAD_IDX = pad_index
        self.SEP_IDX = self.vocab['[SEP]']
        self.CLS_IDX = self.vocab['[CLS]']
        self.MASK_IDS = self.vocab['[MASK]']
        self.batch_size = batch_size
        self.max_sen_len = max_sen_len
        self.max_position_embeddings = max_position_embeddings
        self.pad_index = pad_index
        self.is_sample_shuffle = is_sample_shuffle
        self.data_name = data_name
        self.masked_rate = masked_rate
        self.masked_token_rate = masked_token_rate
        self.masked_token_unchanged_rate = masked_token_unchanged_rate
        self.random_state = random_state
        random.seed(random_state)

    @staticmethod
    def get_format_data(filepath):
        assert filepath == 'ci'
        return readci(filepath)

    @staticmethod
    def get_next_sentence_sample(sentence, next_sentence, paragraphs):
        if random.random() < 0.5:
            is_next = True
        else:
            next_sentence = random.choice(random.choice(paragraphs))
            is_next = False
        return sentence, next_sentence, is_next

    @cache
    def data_process(self, filepath, postfix):
        paragraphs = self.get_format_data(filepath)
        data, max_len = [], 0

        for paragraph in tqdm(paragraphs):  # 遍历每个

            for i in range(len(paragraph) - 1):  # 遍历一个段落中的每一句话

                sentence, next_sentence, is_next = self.get_next_sentence_sample(
                paragraph[i], paragraph[i + 1], paragraphs)  # 构造NSP样本
                token_a_ids = [self.vocab[token] for token in self.tokenizer(sentence)]
                token_b_ids = [self.vocab[token] for token in self.tokenizer(next_sentence)]
                token_ids = [self.CLS_IDX] + token_a_ids + [self.SEP_IDX] + token_b_ids

                if len(token_ids) > self.max_position_embeddings - 1:
                    token_ids = token_ids[:self.max_position_embeddings - 1]
                token_ids += [self.SEP_IDX]
                token_ids = torch.tensor(token_ids, dtype=torch.long)

                seg1 = [0] * (len(token_a_ids) + 2)  # 2 表示[CLS]和中间的[SEP]这两个字符
                seg2 = [1] * (len(token_ids) - len(seg1))
                segs = torch.tensor(seg1 + seg2, dtype=torch.long)

                nsp_lable = torch.tensor(int(is_next), dtype=torch.long)
                max_len = max(max_len, token_ids.size(0))
                data.append([token_ids, segs, nsp_lable])
        all_data = {'data': data, 'max_len': max_len}
        return all_data

    def generate_batch(self, data_batch):
        b_token_ids, b_segs, b_nsp_label, b_mlm_label = [], [], [], []
        for (token_ids, segs, nsp_lable, mlm_label) in data_batch:
            # 开始对一个batch中的每一个样本进行处理
            b_token_ids.append(token_ids)
            b_segs.append(segs)
            b_nsp_label.append(nsp_lable)
            b_mlm_label.append(mlm_label)
        b_token_ids = pad_sequence(b_token_ids,  # [batch_size,max_len]
                                   padding_value=self.PAD_IDX,
                                   batch_first=False,
                                   max_len=self.max_sen_len)
        # b_token_ids:  [src_len,batch_size]

        b_segs = pad_sequence(b_segs,  # [batch_size,max_len]
                              padding_value=self.PAD_IDX,
                              batch_first=False,
                              max_len=self.max_sen_len)
        # b_segs: [src_len,batch_size]

        b_mlm_label = pad_sequence(b_mlm_label,  # [batch_size,max_len]
                                   padding_value=self.PAD_IDX,
                                   batch_first=False,
                                   max_len=self.max_sen_len)
        # b_mlm_label:  [src_len,batch_size]

        b_mask = (b_token_ids == self.PAD_IDX).transpose(0, 1)
        # b_mask: [batch_size,max_len]

        b_nsp_label = torch.tensor(b_nsp_label, dtype=torch.long)
        # b_nsp_label: [batch_size]
        return b_token_ids, b_segs, b_mask, b_mlm_label,

    def load_train_val_test_data(self,
                                 train_file_path=None,
                                 val_file_path=None,
                                 test_file_path=None,
                                 only_test=False):

        postfix = f"_ml{self.max_sen_len}_rs{self.random_state}_mr{str(self.masked_rate)[2:]}" \
                  f"_mtr{str(self.masked_token_rate)[2:]}_mtur{str(self.masked_token_unchanged_rate)[2:]}"

        test_data = self.data_process(filepath=test_file_path, postfix='test' + postfix)['data']
        test_iter = DataLoader(test_data, batch_size=self.batch_size,
                               shuffle=False, collate_fn=self.generate_batch)
        if only_test:
            return test_iter

        data = self.data_process(filepath=train_file_path, postfix='train' + postfix)
        train_data, max_len = data['data'], data['max_len']
        if self.max_sen_len == 'same':
            self.max_sen_len = max_len
        train_iter = DataLoader(train_data, batch_size=self.batch_size,
                                shuffle=self.is_sample_shuffle, collate_fn=self.generate_batch)

        val_data = self.data_process(filepath=val_file_path, postfix='val' + postfix)['data']
        val_iter = DataLoader(val_data, batch_size=self.batch_size,
                              shuffle=False, collate_fn=self.generate_batch)
        return train_iter, test_iter, val_iter
