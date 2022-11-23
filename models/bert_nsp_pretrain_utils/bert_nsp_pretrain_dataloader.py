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
        if record['punc_after'] != 0:
            sentence.append('，' if record['punc_after'] == 1 else '。')
        if record['is_end_in_sentence']:
            paragraph.append(sentence)
            sentence = []
    return paragraphs


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
                 masked_token_unchanged_rate=0.5,
                 nsp_num_classes=2,
                 only_mlm_task=False):
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
        self.nsp_num_classes = nsp_num_classes
        self.only_mlm_task = only_mlm_task
        random.seed(random_state)

    @staticmethod
    def get_format_data(filepath):
        return read_ci(filepath)

    def get_next_phrase_samples(self, paragraph, paragraphs):
        def reconstruct(phrases):
            reconstruction = ''
            for i in range(len(phrases)):
                if phrases[i] == 'sep':
                    continue
                elif phrases[i] == 'com':
                    reconstruction = reconstruction + '，'
                elif phrases[i] == 'stop':
                    reconstruction = reconstruction + '。'
                else:
                    reconstruction = reconstruction + phrases[i]
            return reconstruction

        all_phrases = []
        samples = []
        for sentence in paragraph:
            for p in sentence:
                if p == '，':
                    if all_phrases[-1] == 'sep':
                        all_phrases = all_phrases[:-1]
                    all_phrases.append('com')
                elif p == '。':
                    if all_phrases[-1] == 'sep':
                        all_phrases = all_phrases[:-1]
                    all_phrases.append('stop')
                else:
                    all_phrases.append(p)
                    all_phrases.append('sep')
        if self.only_mlm_task:
            samples.append((reconstruct(all_phrases), '', False))
        else:
            first_half = all_phrases[: len(all_phrases) // 2]
            second_half = all_phrases[len(all_phrases) // 2:]
            reconstruction = reconstruct(first_half)
            prepared_samples = dict(sep=[], com=[], stop=[])
            last_punc = ''
            for p in second_half:
                if p not in ['sep', 'com', 'stop']:
                    if last_punc:
                        prepared_samples[last_punc].append((reconstruction, p))
                    if last_punc == 'sep' or last_punc == '':
                        reconstruction = reconstruction + p
                    elif last_punc == 'com':
                        reconstruction = reconstruction + '，' + p
                    elif last_punc == 'stop':
                        reconstruction = reconstruction + '。' + p
                else:
                    last_punc = p
            if self.nsp_num_classes == 2:
                prepared_samples = {1: prepared_samples['sep'] + prepared_samples['com'] + prepared_samples['stop']}
            elif self.nsp_num_classes == 3:
                prepared_samples = {1: prepared_samples['sep'], 2: prepared_samples['com'] + prepared_samples['stop']}
                min_l = min([len(prepared_samples[1]), len(prepared_samples[2])])
                prepared_samples[1] = prepared_samples[1][:min_l]
                prepared_samples[2] = prepared_samples[2][:min_l]
            elif self.nsp_num_classes == 4:
                prepared_samples = {1: prepared_samples['sep'], 2: prepared_samples['com'], 3: prepared_samples['stop']}
                min_l = min([len(prepared_samples[1]), len(prepared_samples[2]), len(prepared_samples[3])])
                prepared_samples[1] = prepared_samples[1][:min_l]
                prepared_samples[2] = prepared_samples[2][:min_l]
            for (label, sample_list) in prepared_samples.items():
                for item in sample_list:
                    if random.random() > 1/self.nsp_num_classes:
                        samples.append((item[0], item[1], label))
                    else:
                        samples.append((item[0], random.choice(random.choice(random.choice(paragraphs))), 0))
        print(samples)
        input()
        return samples

    def replace_masked_tokens(self, token_ids, candidate_pred_positions, num_mlm_preds):
        """
        本函数的作用是根据给定的token_ids、候选mask位置以及需要mask的数量来返回被mask后的token_ids以及标签信息
        :param token_ids:
        :param candidate_pred_positions:
        :param num_mlm_preds:
        :return:
        """
        pred_positions = []
        mlm_input_tokens_id = [token_id for token_id in token_ids]
        for mlm_pred_position in candidate_pred_positions:
            if len(pred_positions) >= num_mlm_preds:
                break  # 如果已经mask的数量大于等于num_mlm_preds则停止mask
            masked_token_id = None
            # 80%的时间：将词替换为['MASK']词元，但这里是直接替换为['MASK']对应的id
            if random.random() < self.masked_token_rate:  # 0.8
                masked_token_id = self.MASK_IDS
            else:
                # 10%的时间：保持词不变
                if random.random() < self.masked_token_unchanged_rate:  # 0.5
                    masked_token_id = token_ids[mlm_pred_position]
                # 10%的时间：用随机词替换该词
                else:
                    masked_token_id = random.randint(0, len(self.vocab.stoi) - 1)
            mlm_input_tokens_id[mlm_pred_position] = masked_token_id
            pred_positions.append(mlm_pred_position)  # 保留被mask位置的索引信息
        # 构造mlm任务中需要预测位置对应的正确标签，如果其没出现在pred_positions则表示该位置不是mask位置
        # 则在进行损失计算时需要忽略掉这些位置（即为PAD_IDX）；而如果其出现在mask的位置，则其标签为原始token_ids对应的id
        mlm_label = [self.PAD_IDX if idx not in pred_positions
                     else token_ids[idx] for idx in range(len(token_ids))]
        return mlm_input_tokens_id, mlm_label

    def get_masked_sample(self, token_ids):
        """
        本函数的作用是将传入的 一段token_ids的其中部分进行mask处理
        :param token_ids:         e.g. [101, 1031, 4895, 2243, 1033, 10029, 2000, 2624, 1031,....]
        :return: mlm_input_tokens_id:  [101, 1031, 103, 2243, 1033, 10029, 2000, 103,  1031, ...]
                           mlm_label:  [ 0,   0,   4895,  0,    0,    0,    0,   2624,  0,...]
        """
        candidate_pred_positions = []  # 候选预测位置的索引
        for i, ids in enumerate(token_ids):
            # 在遮蔽语言模型任务中不会预测特殊词元，所以如果该位置是特殊词元
            # 那么该位置就不会成为候选mask位置
            if ids in [self.CLS_IDX, self.SEP_IDX]:
                continue
            candidate_pred_positions.append(i)
            # 保存候选位置的索引， 例如可能是 [ 2,3,4,5, ....]
        random.shuffle(candidate_pred_positions)  # 将所有候选位置打乱，更利于后续随机
        # 被掩盖位置的数量，BERT模型中默认将15%的Token进行mask
        num_mlm_preds = max(1, round(len(token_ids) * self.masked_rate))
        mlm_input_tokens_id, mlm_label = self.replace_masked_tokens(
            token_ids, candidate_pred_positions, num_mlm_preds)
        return mlm_input_tokens_id, mlm_label

    @cache
    def data_process(self, filepath, postfix):
        paragraphs = self.get_format_data(filepath)
        data, max_len = [], 0

        for paragraph in tqdm(paragraphs):  # 遍历每个

            for (phrase, next_phrase, is_next) in self.get_next_phrase_samples(paragraph, paragraphs):
                try:
                    token_a_ids = [self.vocab[token] for token in self.tokenizer(phrase)]
                    token_b_ids = [self.vocab[token] for token in self.tokenizer(next_phrase)]
                    token_ids = [self.CLS_IDX] + token_a_ids + [self.SEP_IDX] + token_b_ids
                except:
                    print(next_phrase)
                    continue

                seg1 = [0] * (len(token_a_ids) + 2)  # 2 表示[CLS]和中间的[SEP]这两个字符
                seg2 = [1] * (len(token_b_ids) + 1)
                segs = torch.tensor(seg1 + seg2, dtype=torch.long)

                if len(token_ids) > self.max_position_embeddings - 1:
                    token_ids = token_ids[:self.max_position_embeddings - 1]
                    segs = segs[:self.max_position_embeddings]

                token_ids += [self.SEP_IDX]

                mlm_input_tokens_id, mlm_label = self.get_masked_sample(token_ids)
                token_ids = torch.tensor(mlm_input_tokens_id, dtype=torch.long)
                mlm_label = torch.tensor(mlm_label, dtype=torch.long)
                nsp_lable = torch.tensor(int(is_next), dtype=torch.long)
                max_len = max(max_len, token_ids.size(0))
                data.append([token_ids, segs, mlm_label, nsp_lable])

        all_data = {'data': data, 'max_len': max_len}
        return all_data

    def generate_batch(self, data_batch):
        b_token_ids, b_segs, b_mlm_label, b_nsp_label = [], [], [], []
        for (token_ids, segs, mlm_label, nsp_lable) in data_batch:
            # 开始对一个batch中的每一个样本进行处理
            b_token_ids.append(token_ids)
            b_segs.append(segs)
            b_mlm_label.append(mlm_label)
            b_nsp_label.append(nsp_lable)
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
        b_mlm_label = pad_sequence(b_mlm_label,  # [batch_size,max_len]
                                   padding_value=self.PAD_IDX,
                                   batch_first=True,
                                   max_len=self.max_sen_len)

        b_mask = b_token_ids != self.PAD_IDX
        # b_mask: [batch_size,max_len]

        b_nsp_label = torch.tensor(b_nsp_label, dtype=torch.long)
        # b_nsp_label: [batch_size]
        return b_token_ids, b_segs, b_mask, b_mlm_label, b_nsp_label

    def load_train_val_test_data(self, file_path=None, only_test=False):

        postfix = f"_ml{self.max_sen_len}_rs{self.random_state}_mr{str(self.masked_rate)[2:]}" \
                  f"_mtr{str(self.masked_token_rate)[2:]}_mtur{str(self.masked_token_unchanged_rate)[2:]}" \
                  f"_nspnc{self.nsp_num_classes}_only_mlm{str(self.only_mlm_task)}"

        processed = self.data_process(filepath=file_path, postfix=postfix)

        data, max_len = processed['data'][:], processed['max_len']
        if self.max_sen_len == 'same':
            self.max_sen_len = max_len

        num_samples = len(data)
        num_samples_val, num_samples_test = int(num_samples * self.val_set_portion), \
                                            int(num_samples * self.test_set_portion)
        train_data, val_data, test_data = torch.utils.data.random_split(data,
                                                                        [
                                                                            num_samples - num_samples_test - num_samples_val,
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

    def make_inference_sample(self, history, next_phrase):
        if isinstance(next_phrase, str):
            history = [history]
            next_phrase = [next_phrase]
        input_tokens_ids = []
        for i in range(len(history)):
            history_token_ids = []
            for item in history[i]:
                history_token_ids += [self.vocab[t] for t in self.tokenizer(''.join(item))]
            next_phrase_token_ids = [self.vocab[t] for t in self.tokenizer(next_phrase[i])]
            token_ids = [self.CLS_IDX] + history_token_ids + [self.SEP_IDX] + next_phrase_token_ids
            if len(token_ids) > self.max_position_embeddings - 1:
                token_ids = token_ids[:self.max_position_embeddings - 1]
            token_ids += [self.SEP_IDX]
            input_tokens_ids.append(torch.tensor(token_ids, dtype=torch.long))
        input_tokens_ids = pad_sequence(input_tokens_ids,
                                        padding_value=self.PAD_IDX,
                                        batch_first=True,
                                        max_len=None)
        mask = input_tokens_ids != self.PAD_IDX
        print(input_tokens_ids, mask)
        return input_tokens_ids, mask


if __name__ == '__main__':
    pass
