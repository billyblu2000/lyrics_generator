from typing import List, Union

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

    def __call__(self, phrases: pd.DataFrame, song_structure: SongStructure) -> Lyrics:
        """

        :param phrases:
        :param song_structure:
        """
        pass

    def run(self, phrases: pd.DataFrame, song_structure: SongStructure) -> Lyrics:
        return self(phrases, song_structure)

    def next_phrase_inference(self,
                              history: Union[Lyrics, List[Lyrics]],
                              next_phrase: Union[str, List[str]]) -> (float, float):
        return self._next_phrase_inference(history, next_phrase)

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
        # checkpoint = torch.load(self.config.npp_model_checkpoint_path
        #                         if self.config.npp_model_checkpoint_path
        #                         else self.npp_model_config.model_save_path, map_location=self.npp_model_config.device)
        # loaded_paras = checkpoint['model_state_dict']
        # self.npp_model.load_state_dict(loaded_paras)
        self.npp_model = self.npp_model.to(self.npp_model_config.device)
        self.npp_model.eval()

    def _next_phrase_inference(self,
                               history: Union[Lyrics, List[Lyrics]],
                               next_phrase: Union[str, List[str]]) -> (float, float):
        """

        :param history:
        :param next_phrase:
        """
        token_ids, mask = self.npp_model_data_loader.make_inference_sample(history, next_phrase)
        print(token_ids)
        with torch.no_grad():
            token_ids = token_ids.to(self.npp_model_config.device)
            mask = mask.to(self.npp_model_config.device)
            nsp_logits, nsp_weak_logits = self.npp_model(input_ids=token_ids, attention_mask=mask)
            print(nsp_logits)
            nsp_pred_score = torch.softmax(nsp_logits, dim=-1)[:, 1]
            nsp_weak_pred_score = torch.softmax(nsp_weak_logits, dim=-1)[:, 1]

        return nsp_pred_score, nsp_weak_pred_score

    def _rhyme_loss(self):
        pass

    def _sentence_length_loss(self):
        pass

    def _fluency_loss(self):
        pass


if __name__ == '__main__':
    h = [
        [('一阳才动', '，'), ('万物生春意', '。'), ('试与问宫梅', '，'), ('到东阁、花枝第几', '。')],
        [('一阳才动', '，'), ('万物生春意', '。'), ('试与问宫梅', '，'), ('到东阁、花枝第几', '。')],
        [('一阳才动', '，'), ('万物生春意', '。'), ('试与问宫梅', '，'), ('到东阁、花枝第几', '。')],
        [('一阳才动', '，'), ('万物生春意', '。'), ('试与问宫梅', '，'), ('到东阁、花枝第几', '。')]
    ]
    n = [
        '疏疏淡淡',
        '恨中秋',
        '箫韶妙曲',
        '辔摇衔铁'
    ]
    connector = PhraseConnector(config)
    print(connector.next_phrase_inference(h, n))
