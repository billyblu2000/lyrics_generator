import random

import pandas
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import T_co
from tqdm import tqdm
from transformers import BertTokenizer

from models.bert_nsp_pretrain_utils.bert_nsp_pretrain_config import ModelConfig
from models.bert_nsp_pretrain_utils.bert_nsp_pretrain_utils import cache, build_vocab, pad_sequence


def read_ci(filepath='data/phrase_database_d0_drop_dup.csv'):
    ci = pandas.read_csv(filepath, index_col=0)
    paragraphs, paragraph, sentence = [], [], []
    last_song_id = -1
    for i in range(len(ci['phrase'])):
        record = ci.loc[i]
        if record['song_index'] != last_song_id:
            if paragraph:
                paragraphs.append(paragraph)
            paragraph = []
            sentence = []
            last_song_id = record['song_index']
        sentence.append(record['phrase'])
        if record['is_end_in_sentence']:
            paragraph.append(sentence)
            sentence = []
    return paragraphs


class CiDataset(Dataset):

    def __init__(self, dataset):
        self.data = dataset['data']
        self.max_len = dataset['max_len']

    def __getitem__(self, index) -> T_co:
        return self.data[index]

    def __len__(self):
        return len(self.data)


class LoadBertPretrainingDataset(object):
    def __init__(self,
                 vocab_path='./vocab.txt',
                 tokenizer=None,
                 batch_size=32,
                 max_sen_len=None,
                 max_position_embeddings=512,
                 pad_index=0,
                 is_sample_shuffle=True,
                 val_set_portion=0.1,
                 test_set_portion=0.1,
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
        self.val_set_portion = val_set_portion
        self.test_set_portion = test_set_portion
        self.data_name = data_name
        self.masked_rate = masked_rate
        self.masked_token_rate = masked_token_rate
        self.masked_token_unchanged_rate = masked_token_unchanged_rate
        self.random_state = random_state
        random.seed(random_state)

    @staticmethod
    def get_format_data(filepath):
        return read_ci(filepath)

    @staticmethod
    def get_next_phrase_samples(paragraph, paragraphs):
        all_phrases = []
        samples = []
        for sentence in paragraph:
            all_phrases += sentence
            all_phrases.append('sep')
        for i in range(len(all_phrases) - 2):
            if all_phrases[i] == 'sep':
                continue
            if all_phrases[i + 1] != 'sep':
                next_phrase = all_phrases[i + 1]
                is_next, is_next_weak = True, False
            else:
                next_phrase = all_phrases[i + 2]
                is_next, is_next_weak = False, True
            if random.random() < 0.5:
                next_phrase = random.choice(random.choice(random.choice(paragraphs)))
                is_next, is_next_weak = False, False
            samples.append((all_phrases[i], next_phrase, is_next, is_next_weak))
        return samples

    @cache
    def data_process(self, filepath, postfix):
        paragraphs = self.get_format_data(filepath)
        data, max_len = [], 0

        for paragraph in tqdm(paragraphs):  # 遍历每个

            for (phrase, next_phrase, is_next, is_next_weak) in self.get_next_phrase_samples(paragraph, paragraphs):
                try:
                    token_a_ids = [self.vocab[token] for token in self.tokenizer(phrase)]
                    token_b_ids = [self.vocab[token] for token in self.tokenizer(next_phrase)]
                    token_ids = [self.CLS_IDX] + token_a_ids + [self.SEP_IDX] + token_b_ids
                except:
                    print(next_phrase)
                    continue

                if len(token_ids) > self.max_position_embeddings - 1:
                    token_ids = token_ids[:self.max_position_embeddings - 1]
                token_ids += [self.SEP_IDX]
                token_ids = torch.tensor(token_ids, dtype=torch.long)

                seg1 = [0] * (len(token_a_ids) + 2)  # 2 表示[CLS]和中间的[SEP]这两个字符
                seg2 = [1] * (len(token_ids) - len(seg1))
                segs = torch.tensor(seg1 + seg2, dtype=torch.long)

                nsp_lable = torch.tensor(int(is_next), dtype=torch.long)
                nsp_label_weak = torch.tensor(int(is_next_weak), dtype=torch.long)
                max_len = max(max_len, token_ids.size(0))
                data.append([token_ids, segs, nsp_lable, nsp_label_weak])
        all_data = {'data': data, 'max_len': max_len}
        return all_data

    def generate_batch(self, data_batch):
        b_token_ids, b_segs, b_nsp_label, b_nsp_label_weak = [], [], [], []
        for (token_ids, segs, nsp_lable, nsp_label_weak) in data_batch:
            # 开始对一个batch中的每一个样本进行处理
            b_token_ids.append(token_ids)
            b_segs.append(segs)
            b_nsp_label.append(nsp_lable)
            b_nsp_label_weak.append(nsp_label_weak)
        b_token_ids = pad_sequence(b_token_ids,  # [batch_size,max_len]
                                   padding_value=self.PAD_IDX,
                                   batch_first=True,
                                   max_len=self.max_sen_len)
        # b_token_ids:  [src_len,batch_size]

        b_segs = pad_sequence(b_segs,  # [batch_size,max_len]
                              padding_value=self.PAD_IDX,
                              batch_first=True,
                              max_len=self.max_sen_len)
        # b_segs: [src_len,batch_size]

        b_mask = b_token_ids != self.PAD_IDX
        # b_mask: [batch_size,max_len]

        b_nsp_label = torch.tensor(b_nsp_label, dtype=torch.long)
        b_nsp_label_weak = torch.tensor(b_nsp_label_weak, dtype=torch.long)
        # b_nsp_label: [batch_size]
        return b_token_ids, b_segs, b_mask, b_nsp_label, b_nsp_label_weak

    def load_train_val_test_data(self, file_path=None, only_test=False):

        postfix = f"_ml{self.max_sen_len}_rs{self.random_state}_mr{str(self.masked_rate)[2:]}" \
                  f"_mtr{str(self.masked_token_rate)[2:]}_mtur{str(self.masked_token_unchanged_rate)[2:]}"

        processed = self.data_process(filepath=file_path, postfix=postfix)

        data, max_len = processed['data'][:], processed['max_len']
        if self.max_sen_len == 'same':
            self.max_sen_len = max_len

        num_samples = len(data)
        num_samples_val, num_samples_test = int(num_samples * self.val_set_portion), \
                                            int(num_samples * self.test_set_portion)
        train_data, val_data, test_data = torch.utils.data.random_split(data,
                                                                        [num_samples-num_samples_test-num_samples_val,
                                                                        num_samples_val, num_samples_test])

        test_iter = DataLoader(test_data, batch_size=self.batch_size,
                               shuffle=False, collate_fn=self.generate_batch)
        if only_test:
            return test_iter

        train_iter = DataLoader(train_data, batch_size=self.batch_size,
                                shuffle=self.is_sample_shuffle, collate_fn=self.generate_batch)
        val_iter = DataLoader(val_data, batch_size=self.batch_size,
                              shuffle=False, collate_fn=self.generate_batch)
        return train_iter, test_iter, val_iter


if __name__ == '__main__':
    config = ModelConfig()
    bert_tokenize = BertTokenizer.from_pretrained(config.pretrained_model_dir).tokenize
    data_loader = LoadBertPretrainingDataset(vocab_path=config.vocab_path,
                                             tokenizer=bert_tokenize,
                                             batch_size=config.batch_size,
                                             max_sen_len=config.max_sen_len,
                                             max_position_embeddings=config.max_position_embeddings,
                                             pad_index=config.pad_index,
                                             is_sample_shuffle=config.is_sample_shuffle,
                                             validation_set_portion=config.validation_set_portion,
                                             test_set_portion=config.test_set_portion,
                                             random_state=config.random_state,
                                             data_name=config.data_name,
                                             masked_rate=config.masked_rate,
                                             masked_token_rate=config.masked_token_rate,
                                             masked_token_unchanged_rate=config.masked_token_unchanged_rate)
    data_loader.data_process(filepath=config.dataset_dir, postfix='bert_nsp_ci_pretrain')
