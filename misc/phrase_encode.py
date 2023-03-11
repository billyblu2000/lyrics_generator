import json

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

import config


def encode_phrase_granu(phrases):
    phrase_list = list(phrases['phrase'])
    embs = []
    for i in tqdm(range(0, 310000, 10000)):
        embs.append(st.encode(phrase_list[i:min([i + 10000, len(phrase_list)])]))
    np.savez_compressed('phrase_embs', np.concatenate(embs))


def encode_sentence(all_sentences):
    print(all_sentences[:10])
    embs = []
    for i in tqdm(range(0, len(all_sentences), 10000)):
        embs.append(st.encode(all_sentences[i:min([i + 10000, len(all_sentences)])]))
    np.savez_compressed('sentence_embs', np.concatenate(embs))


def encode_sentence_granu(phrases):
    current_song_index = -1
    current_sentence = -1
    current_concat = ''
    count_phrase = 1
    embs = []
    phrase_list = list(phrases['phrase'])
    song_index_list = list(phrases['song_index'])
    sentence_index_list = list(phrases['sentence_position_in_song'])
    for i in tqdm(range(len(phrase_list))):
        try:
            phrase, song_index, sentence_index = phrase_list[i], song_index_list[i], sentence_index_list[i]
            if song_index == current_song_index and sentence_index == current_sentence:
                current_concat = current_concat + phrase
                count_phrase += 1
            else:
                if current_concat != '':
                    try:
                        emb = np.array([st.encode(current_concat)])
                    except:
                        emb = np.array([np.zeros(512)])
                    embs.append(np.repeat(emb, count_phrase, axis=0))
                current_concat = phrase
                count_phrase = 1
                current_song_index, current_sentence = song_index, sentence_index
        except:
            embs = np.concatenate(embs)
            np.savez_compressed('phrase_embs_sentence_granu', embs)
            return
    embs = np.concatenate(embs)
    print(len(embs))
    np.savez_compressed('phrase_embs_sentence_granu', embs)


if __name__ == '__main__':
    st = SentenceTransformer(config.sentence_transformer_path)
    # phrases = pd.read_csv('../data/phrase_database_d0_drop_dup.csv', index_col=0, encoding="utf-8")
    # encode_sentence_granu(phrases)
    sentences = json.load(open('../data/all_sentence.json', 'r'))
    encode_sentence(sentences)
