import random

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class SentenceTransformersRetriever:

    def __init__(self, config, retrieve_num=None):
        self.config = config
        if self.config.print_log:
            print('Init Model: SentenceTransformersRetriever...')
        self.retrieve_num = retrieve_num if retrieve_num else config.retrieve_num
        self.st_model = SentenceTransformer(self.config.sentence_transformer_path)
        self.phrase_embedding = np.load(self.config.phrase_embedding)['arr_0']
        self.phrase_database = pd.read_csv(self.config.phrase_database, index_col=0)

    def __call__(self, title: str) -> pd.DataFrame:
        """

        :param title: the prompt
        :return: pandas DataFrame containing the retrieved phrases
        """
        # load model and data
        if self.config.print_log:
            print(f'Start retrieving with prompt "{title}"...')

        # encode title
        title_embedding = self.st_model.encode(title)

        # compare sim
        sims = cosine_similarity(self.phrase_embedding, title_embedding.reshape(1, -1))

        # find k largest
        sims_with_index = [(i, sims[i]) for i in range(len(sims))]
        # selected = heapq.nlargest(self.retrieve_num, sims_with_index, key=lambda x: x[1])
        selected = self.__sample_from_sorted_retrieve(sorted(sims_with_index, key=lambda x:x[1], reverse=True))
        selected_index = [i[0] for i in selected]
        return self.phrase_database.loc[[i in selected_index for i in range(len(self.phrase_database))]]

    def __sample_from_sorted_retrieve(self, lst):
        """

        :param lst:
        :return:
        """
        samples_in_each_interval = 10
        interval_length = 500
        num_intervals = self.retrieve_num // samples_in_each_interval
        sampled = []
        for i in range(num_intervals):
            sampled += random.sample(lst[i*interval_length: i*interval_length + interval_length],
                                     k=samples_in_each_interval)
        return sampled
