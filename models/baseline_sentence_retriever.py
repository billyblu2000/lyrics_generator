import json
import random

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class BaselineSentenceRetriever:

    def __init__(self, config):
        self.config = config
        self.retrieve_num = 1000
        self.st_model = SentenceTransformer(self.config.sentence_transformer_path)
        self.embedding = np.load(config.baseline_sentence_embedding)['arr_0']
        self.database = json.load(open(config.baseline_sentence_database, 'r'))

    def __call__(self, title: str):
        """

        :param title: the prompt
        :return: pandas DataFrame containing the retrieved phrases
        """
        # encode title
        title_embedding = self.st_model.encode(title)

        # compare sim
        sims = cosine_similarity(self.embedding, title_embedding.reshape(1, -1))

        # find k largest
        sims_with_index = [(i, sims[i]) for i in range(len(sims))]
        # selected = heapq.nlargest(self.retrieve_num, sims_with_index, key=lambda x: x[1])
        selected = self.__sample_from_sorted_retrieve(sorted(sims_with_index, key=lambda x: x[1], reverse=True))
        selected_index = [i[0] for i in selected]
        all_sentences = [self.database[i] for i in selected_index]
        all_sentences_len_map = {4: [], 5: [], 6: [], 7: []}
        for i in all_sentences:
            l = len(i)
            if l in all_sentences_len_map:
                all_sentences_len_map[l].append(i)
        return all_sentences_len_map

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
            sampled += random.sample(lst[i * interval_length: i * interval_length + interval_length],
                                     k=samples_in_each_interval)
        return sampled


if __name__ == '__main__':
    pass
