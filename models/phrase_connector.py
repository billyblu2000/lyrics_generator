
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
        pass

    def run(self, phrases: pd.DataFrame, song_structure: SongStructure) -> Lyrics:
        return self(phrases, song_structure)

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
        self.all_rhymes = ['a', 'o', 'e', 'e2', 'ai', 'ei', 'ao', 'ou', 'an', 'en', 'ang', 'eng', 'er']
        self.num_rhymes = len(self.all_rhymes)
        self.rhyme_vector_mapping = {}
        for i in range(len(self.all_rhymes)):
            self.rhyme_vector_mapping[self.all_rhymes[i]] = np.zeros(self.num_rhymes)
            self.rhyme_vector_mapping[self.all_rhymes[i]][i] = 1
        self.max_rhyme_variance = np.var([0]*(self.num_rhymes-1) + [1])*self.num_rhymes

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
            nsp_pred_score = np.array(torch.softmax(nsp_logits, dim=-1))

        return nsp_pred_score

    def _rhyme_loss(self, history: Union[Lyrics, List[Lyrics]],
                    next_phrase: Union[str, List[str]]) -> Union[List[float], float]:
        """

        :rtype: object
        :param history:
        :param next_phrase:
        """
        all_rhyme_words = []

    def _sentence_length_loss(self, history: Union[Lyrics, List[Lyrics]],
                              next_phrase: Union[str, List[str]],
                              sentence_structure: SongStructure) -> Union[List[float], float]:
        """

        :rtype: object
        :param history:
        :param next_phrase:
        :param sentence_structure:
        """
        pass

    def _fluency_loss(self, history: Union[Lyrics, List[Lyrics]],
                      next_phrase: Union[str, List[str]]) -> Union[List[float], float]:
        """

        :rtype: object
        :param history:
        :param next_phrase:
        """
        pass


if __name__ == '__main__':
    all_phrases = open('test', 'r', encoding='utf-8').readline().split(',')
    all_phrases = [i.strip().rstrip("'").lstrip("'") for i in all_phrases]
    print(all_phrases)
    connector = PhraseConnector(config)
    for i in range(100):
        prev_p = random.choice(all_phrases)
        next_p = random.choice(all_phrases)
        print(f'{prev_p} {next_p}: {connector.next_phrase_inference([(prev_p)], next_p)}',)
