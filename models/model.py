import random
import time
from typing import List

from models.baseline_sentence_retriever import BaselineSentenceRetriever
from models.phrase_connector import PhraseConnector
from models.sentence_transformer_retriever import SentenceTransformersRetriever
import config
from models.types import SongStructure


class Model:

    def __init__(self, config, retriever, connector):
        """

        :param config:
        :param retriever:
        :param connector:
        """
        if config.print_log:
            print('Init Model: Model...')
        self.config = config
        self.retriever = retriever
        self.connector = connector

    @classmethod
    def init_model(cls):
        """

        :return:
        """
        retriever = SentenceTransformersRetriever(config)
        connector = PhraseConnector(config)
        return Model(config, retriever, connector)

    def run(self, rhythmic: str, title: str):
        """

        :param rhythmic:
        :param title:
        """
        if config.print_log:
            print(f'Running model with rhythmic "{rhythmic}" and title "{title}"...')
        phrases = self.retriever(title)
        # print(list(phrases[0]['phrase']), list(phrases[1]['phrase']))
        song_structure = self.find_song_structure(rhythmic)
        lyrics = self.connector(phrases, song_structure)
        print(lyrics)
        file = open(f'{title}-{time.time()}.txt', 'w')
        file.write(lyrics)
        file.close()

    def __call__(self, rhythmic: str, title: str):
        return self.run(rhythmic, title)

    @staticmethod
    def find_song_structure(rhythmic: str) -> SongStructure:
        """

        :param rhythmic:
        """
        test = [
            ([(5, '，'), (5, '。'), (7, '，'), (5, '。'), (5, '，'), (5, '。'), (7, '，'), (5, '。')], '卜算子'),
            ([(6, '，'), (6, '。'), (7, '。'), (6, '。'), (6, '，'), (6, '。'), (7, '。'), (6, '。')], '西江月'),
            ([(4, '，'), (4, '。'), (7, '。'), (7, '，'), (7, '。'), (4, '，'), (4, '。'), (7, '。'), (7, '，'), (7, '。'),], '踏莎行'),
            ([(7, '。'), (4, '，'), (5, '。'), (7, '，'), (7, '。'), (7, '。'), (4, '，'), (5, '。'), (7, '，'), (7, '。'), ],
             '蝶恋花')
        ]
        return test[random.randint(0,3)][0]


class Baseline:

    def __init__(self, config, retriever):
        """

        :param config:
        :param retriever:
        :param connector:
        """
        if config.print_log:
            print('Init Model: Model...')
        self.config = config
        self.retriever = retriever

    @classmethod
    def init_model(cls):
        retriever = BaselineSentenceRetriever(config)
        return Baseline(config, retriever)

    def run(self, rhythmic: str, title: str):
        return self(rhythmic, title)

    def __call__(self, rhythmic: str, title: str):
        song_structure = self.find_song_structure(rhythmic)
        sentences = self.retriever(title)
        lyrics = ''
        for i in song_structure:
            lyrics = lyrics + random.choice(sentences[i[0]]) + i[1]
        print(lyrics)
        return lyrics

    @staticmethod
    def find_song_structure(rhythmic: str) -> SongStructure:
        """

        :param rhythmic:
        """
        test = [
            # ([(5, '，'), (5, '。'), (7, '，'), (5, '。'), (5, '，'), (5, '。'), (7, '，'), (5, '。')], '卜算子'),
            ([(6, '，'), (6, '。'), (7, '。'), (6, '。'), (6, '，'), (6, '。'), (7, '。'), (6, '。')], '西江月'),
            ([(4, '，'), (4, '。'), (7, '。'), (7, '，'), (7, '。'), (4, '，'), (4, '。'), (7, '。'), (7, '，'), (7, '。'), ],
             '踏莎行'),
            # ([(7, '。'), (4, '，'), (5, '。'), (7, '，'), (7, '。'), (7, '。'), (4, '，'), (5, '。'), (7, '，'), (7, '。'), ],
            #  '蝶恋花')
        ]
        return random.choice(test)[0]
