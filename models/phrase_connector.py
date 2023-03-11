# coding=gbk
import copy
import random
from typing import List, Union, Sequence

import numpy as np
import pandas as pd
import torch
from transformers import BertTokenizer

import config
from models.bert_nsp_pretrain_utils.bert_nsp_pretrain_dataloader import LoadBertPretrainingDataset
from models.bert_nsp_pretrained import BertNSPPretrained
from models.bert_nsp_pretrain_utils.bert_nsp_pretrain_config import ModelConfig
from models.types import SongStructure, Lyrics


class PhraseConnector:

    def __init__(self, config):
        """

        :param config:
        """
        self.config = config
        self.__next_phrase_predictor_model_init()
        self.__rhyme_model_init()

    def __call__(self, phrases: pd.DataFrame, song_structure: SongStructure) -> Lyrics:
        """

        :rtype: Lyrics
        :param phrases:
        :param song_structure:
        """
        print('Start connecting...')
        return self._search(phrases, song_structure)

    def run(self, phrases: pd.DataFrame, song_structure: SongStructure) -> Lyrics:
        return self(phrases, song_structure)

    def _search(self, phrases: pd.DataFrame, song_structure: SongStructure) -> Lyrics:

        def loss(fl, rl, sl):
            return fl + 2 * rl + 10 * sl

        def is_complete(l):
            return len(l) == len(song_structure) and len(l[-1][0]) - song_structure[-1][0] >= -1

        first_half_phrases = phrases[0]
        second_half_phrases = phrases[1]
        begin_phrases = list(
            first_half_phrases.loc[first_half_phrases['is_begin_in_sentence'] == 1].loc[first_half_phrases['sentence_position_in_song'] == 0]['phrase'])
        all_phrases_first_half = list(first_half_phrases['phrase'])
        all_phrases_second_half = list(second_half_phrases['phrase'])
        next_iter = [[[(p,)], 0, [p]] for p in begin_phrases]
        completed = []
        num_beams = 5

        while next_iter:
            this_iter = copy.deepcopy(next_iter)
            next_iter = []
            print('this_iter')
            # print(this_iter)
            # go through this iteration, add potential next phrase to next iteration
            for item in this_iter:
                # print('item', item)
                history = item[0]
                candidates = []
                temp = []
                if len(item[0]) < len(song_structure) // 2:
                    all_phrases = all_phrases_first_half
                else:
                    all_phrases = all_phrases_second_half
                for next_phrase in all_phrases:
                    if next_phrase in item[2]:
                        continue
                    else:
                        temp.append(next_phrase)
                j = 0
                all_fluency_loss = []
                while j*100 < len(temp):
                    all_fluency_loss += list(self._fluency_loss(history=[history] * len(temp[j*100: (j+1)*100]), next_phrase=temp[j*100: (j+1)*100]))
                    j += 1
                all_fluency_loss = np.array(all_fluency_loss)
                # print(all_fluency_loss)
                count = 0
                for next_phrase in temp:
                    possibilities = [
                        history[:-1] + [(history[-1][0] + next_phrase,)],
                        history[:-1] + [(history[-1][0], '，'), (next_phrase,)],
                        history[:-1] + [(history[-1][0], '。'), (next_phrase,)],
                    ]
                    fluency_loss = all_fluency_loss[count]
                    rhyme_loss = self._rhyme_loss(possibilities)
                    sentence_length_loss = self._sentence_length_loss(possibilities, song_structure)
                    for i in range(len(possibilities)):
                        with_loss = [possibilities[i],
                                     loss(fluency_loss[i], rhyme_loss[i], sentence_length_loss[i]),
                                     item[2] + [next_phrase]]
                        if is_complete(possibilities[i]):
                            completed.append(with_loss)
                        elif len(possibilities[i]) <= len(song_structure):
                            candidates.append(with_loss)
                    count += 1
                candidates = sorted(candidates, key=lambda x: x[1])[:num_beams]
                next_iter += candidates

            # prune next iteration
            next_iter = sorted(next_iter, key=lambda x: x[1])[:len(begin_phrases) * num_beams]
        completed = sorted(completed, key=lambda x: x[1])
        return completed[0][0]

    def __next_phrase_predictor_model_init(self):
        """
        Initialize the npp model
        """
        self.npp_model_config = ModelConfig()
        self.npp_model_tokenize = BertTokenizer.from_pretrained(self.npp_model_config.pretrained_model_dir).tokenize
        self.npp_model_data_loader = LoadBertPretrainingDataset(vocab_path=self.npp_model_config.vocab_path,
                                                                tokenizer=self.npp_model_tokenize,
                                                                pad_index=self.npp_model_config.pad_index,
                                                                random_state=self.npp_model_config.random_state,
                                                                masked_rate=0.15)
        self.npp_model = BertNSPPretrained(self.npp_model_config, self.npp_model_config.pretrained_model_dir)
        checkpoint = torch.load(self.config.npp_model_checkpoint_path
                                if self.config.npp_model_checkpoint_path
                                else self.npp_model_config.model_save_path, map_location=self.npp_model_config.device)
        loaded_paras = checkpoint['model_state_dict']
        self.npp_model.load_state_dict(loaded_paras)
        self.npp_model = self.npp_model.to(self.npp_model_config.device)
        self.npp_model.eval()

    def __rhyme_model_init(self):
        self.rhyme_database = pd.read_csv(self.config.rhyme_lookup_table)
        self.all_rhymes = ['a', 'o', 'e', 'e2', 'ai', 'ei', 'ao', 'ou', 'an',
                           'en', 'ang', 'eng', 'er', 'i', 'i2', 'u', 'u2']
        self.num_rhymes = len(self.all_rhymes)
        self.rhyme_vector_mapping = {}
        self.word_rhyme_mapping = {self.rhyme_database.loc[i]['word']: self.rhyme_database.loc[i]['rhyme']
                                   for i in range(len(self.rhyme_database['word']))}
        for i in range(len(self.all_rhymes)):
            self.rhyme_vector_mapping[self.all_rhymes[i]] = np.zeros(self.num_rhymes)
            self.rhyme_vector_mapping[self.all_rhymes[i]][i] = 1
        self.max_rhyme_variance = np.var([0] * (self.num_rhymes - 1) + [1]) * self.num_rhymes

    def next_phrase_inference(self,
                              history: Union[Lyrics, List[Lyrics]],
                              next_phrase: Union[str, List[str]]) -> Sequence:
        return self._next_phrase_inference(history, next_phrase)

    def _next_phrase_inference(self,
                               history: Union[Lyrics, List[Lyrics]],
                               next_phrase: Union[str, List[str]]) -> Sequence:
        """

        :rtype: numpy array with shape (batch, 3)
        :param history: Lyrics history or sequence of Lyrics history
        :param next_phrase: next phrase string or sequence of next phrase string
        """
        token_ids, mask = self.npp_model_data_loader.make_inference_sample(history, next_phrase)
        with torch.no_grad():
            token_ids = token_ids.to(self.npp_model_config.device)
            mask = mask.to(self.npp_model_config.device)
            _, nsp_logits = self.npp_model(input_ids=token_ids, attention_mask=mask)
            nsp_pred_score = np.array(torch.softmax(nsp_logits, dim=-1).cpu())

        return nsp_pred_score

    def _rhyme_loss(self, history: Union[Lyrics, List[Lyrics]]) -> Union[List[float], float]:
        """

        :rtype: rhyme loss or a list of rhyme losses, depending on the input shape
        :param history: (a list of) the generated lyrics history
        """

        if isinstance(history[0], tuple):
            history = [history]
        all_losses = []
        for i in range(len(history)):
            all_rhyme_words = []
            for item in history[i]:
                if len(item) == 2:
                    if item[1] == '。':
                        all_rhyme_words.append(item[0][-1])
            all_rhyme_vectors = []
            for w in all_rhyme_words:
                try:
                    all_rhyme_vectors.append(self.rhyme_vector_mapping[self.word_rhyme_mapping[w]])
                except:
                    pass
            all_rhyme_vectors = np.array(all_rhyme_vectors)
            if len(all_rhyme_vectors) != 0:
                all_losses.append(np.var(all_rhyme_vectors, axis=0).sum() / self.max_rhyme_variance)
            else:
                all_losses.append(0.)
        if len(all_losses) == 1:
            return all_losses[0]
        else:
            return all_losses

    def _sentence_length_loss(self, history: Union[Lyrics, List[Lyrics]],
                              sentence_structure: SongStructure) -> Union[List[float], float]:
        """

        :rtype: object
        :param history:
        :param sentence_structure:
        """
        if isinstance(history[0], tuple):
            history = [history]
        all_losses = []
        for i in range(len(history)):
            loss = 0
            # print(history)
            for j in range(len(history[i])):
                # print(j)
                act_length = len(history[i][j][0])
                # print(act_length)
                try:
                    exp_length = sentence_structure[j][0]
                    # print(exp_length)
                    loss += self._single_sentence_length_loss(exp_length, act_length)
                    if sentence_structure[j][1] != history[i][j][1]:
                        loss += 1
                    # print(loss)
                except:
                    # print(j)
                    loss += 1
            loss /= len(history[i])
            all_losses.append(loss)
        if len(all_losses) == 1:
            return all_losses[0]
        else:
            return all_losses

    def _position_loss(self):
        pass

    @staticmethod
    def _single_sentence_length_loss(exp: int, act: int) -> float:
        x = abs(exp - act) / exp
        if x < 1:
            return (2 - x) * x
        else:
            return x

    def _fluency_loss(self, history: Union[Lyrics, List[Lyrics]],
                      next_phrase: Union[str, List[str]]) -> Union[List[float], float]:
        """

        :rtype: object
        :param history:
        :param next_phrase:
        """
        if type(next_phrase) == str:
            logits = self._next_phrase_inference(history=history, next_phrase=next_phrase)[0][1:]
            return 1 - logits
        else:
            logits = self._next_phrase_inference(history=history, next_phrase=next_phrase)[:, 1:]
            return 1 - logits


if __name__ == '__main__':
    phrase = ['东风半夜度关山，和雪到阑干。', '怪见', '梅梢', '未暖', '情知', '柳眼', '犹寒', '青丝菜甲', '银泥饼饵',
              '随分杯盘', '已把', '宜春', '缕胜', '更将', '长命', '题幡']

    connector = PhraseConnector(config)
    result = phrase[0]
    for i in phrase[1:]:
        sm = connector.next_phrase_inference([(result)], i)
        max_index = np.argmax(sm[0][1:]) + 1

        if max_index == 1:
            result = result + i
        elif max_index == 2:
            result = result + '，' + i
        elif max_index == 3:
            result = result + '。' + i
    print(result)
