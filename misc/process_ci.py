import json
import os
import hanlp
import pandas as pd
from tqdm import tqdm

HanLP = hanlp.load('../data/hanlp/mtl/ud_ontonotes_tok_pos_lem_fea_ner_srl_dep_sdp_con_xlm_base_20220608_003435')


def extract_rhythmic_and_title(name):
    if '・' in name:
        r = name.split('・')[0]
        t = name.split('・')[1]
    else:
        r = name
        t = None
    return r, t


def extract_paragraphs(cip):
    paragraphs = []
    punc = []
    sentences = []
    for p in cip:
        temp = p.split('，')
        for t in temp:
            paragraphs += t.split('、')
    for i in paragraphs:
        if i[-1] == '。':
            punc.append(2)
            sentences.append(i[:-1])
        else:
            punc.append(1)
            sentences.append(i)
    return sentences, punc


def get_index(frame, q):
    index = frame[frame.name == q].index.to_list()
    assert len(index) == 0 or len(index) == 1
    if len(index) == 0:
        frame.loc[len(frame)] = [q]
        return len(frame) - 1
    else:
        return index[0]


def process_con(con, depth, dynamic_depth_threshold=4, start_indicator=True, end_indicator=True):
    result = []
    while len(con) == 1 and con.height() > 2:
        con = con[0]
    if con.height() == 2:
        result = [(con[0], con.label(), start_indicator, end_indicator)]
        return result
    else:
        if depth > 1:
            for i in range(len(con)):
                result += process_con(con[i], depth=depth - 1, start_indicator=start_indicator and i == 0,
                                      end_indicator=end_indicator and i == len(con) - 1)
        elif depth == 0:
            phrase = ''.join(con.leaves())
            if len(phrase) > dynamic_depth_threshold:

                for i in range(len(con)):
                    result += process_con(con[i], 0, start_indicator=start_indicator and i == 0,
                                          end_indicator=end_indicator and i == len(con) - 1)
            else:
                result += [(phrase, con.label(), start_indicator, end_indicator)]
        else:
            for i in range(len(con)):
                phrase = ''.join(con[i].leaves())
                label = con[i].label()
                result += [(phrase, label, start_indicator and i == 0, end_indicator and i == len(con) - 1)]
    return result


def add_song(rhy_index, author_index, title, song_index, cons, punc, depth):
    counter = 0
    song_phrases = []
    for con in cons:
        phrases = process_con(con, depth=depth)
        for item in phrases:
            punc_after = 0
            if item[3]:
                punc_after = punc.pop(0)
            phrase_dict = {
                'phrase': item[0],
                'type': item[1],
                'is_begin_in_sentence': 1 if item[2] else 0,
                'is_end_in_sentence': 1 if item[3] else 0,
                'sentence_position_in_song': counter,
                'rhy_index': rhy_index,
                'author_index': author_index,
                'song_index': song_index,
                'song_title': title,
                'punc_after': punc_after
            }
            song_phrases.append(phrase_dict)
        counter += 1
    return song_phrases


def main(depth=0, overwrite_ar_index=False, save_path='../data/'):
    database = []
    if 'author_index.csv' in os.listdir(save_path) and not overwrite_ar_index:
        author_frame = pd.read_csv(os.path.join(save_path, 'author_index.csv'), index_col=0)
    else:
        author_frame = pd.DataFrame(columns=['name'])
    if 'rhythmic_index.csv' in os.listdir(save_path) and not overwrite_ar_index:
        rhythmic_frame = pd.read_csv(os.path.join(save_path, 'rhythmic_index.csv'), index_col=0)
    else:
        rhythmic_frame = pd.DataFrame(columns=['name'])

    counter = 0
    for file in os.listdir('../data/ci'):
        if 'ci.song' not in file:
            continue
        print(f'Processing {file}')
        data = json.load(open(os.path.join('../data/ci', file), 'r', encoding='utf-8'))
        for ci in tqdm(data):
            rhy, title = extract_rhythmic_and_title(ci['rhythmic'])
            author = ci['author']
            rhy_index = get_index(rhythmic_frame, rhy)
            author_index = get_index(author_frame, author)
            try:
                paragraphs, punc = extract_paragraphs(ci['paragraphs'])
                database += add_song(rhy_index, author_index, title, counter, HanLP(paragraphs)['con'], punc,
                                     depth=depth)
                counter += 1
            except:
                print('An error occurred!')
                continue

    database = pd.DataFrame(database, columns=['phrase', 'type', 'is_begin_in_sentence', 'is_end_in_sentence',
                                               'sentence_position_in_song', 'rhy_index',
                                               'author_index', 'song_index', 'song_title', 'punc_after'])
    database.to_csv(os.path.join(save_path, f'phrase_database_d{depth}.csv'))
    rhythmic_frame.to_csv(os.path.join(save_path, 'rhythmic_index.csv'))
    author_frame.to_csv(os.path.join(save_path, 'author_index.csv'))


def cut_sentence():
    all_s = []
    for file in os.listdir('../data/ci'):
        if 'ci.song' not in file:
            continue
        print(f'Processing {file}')
        data = json.load(open(os.path.join('../data/ci', file), 'r', encoding='utf-8'))

        for ci in tqdm(data):
            p = ci['paragraphs']
            for sentence in p:
                if '、' in sentence:
                    continue
                sentence = sentence.rstrip('。')
                sentence = sentence.split("，")
                for s in sentence:
                    all_s.append(s)
    file = open('../data/all_sentence.json', 'w', encoding='utf-8')
    json.dump(all_s, file)
    file.close()


if __name__ == '__main__':
    # main()
    cut_sentence()
