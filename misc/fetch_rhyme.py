import time

import pandas as pd
from bs4 import BeautifulSoup
import requests


def get_words(rhyme):
    print(rhyme)
    all_words = []
    for tone in range(5):
        res = requests.get(f'https://infoqme.com/yungcn/word/all-sp-sp-s{tone + 1}.html', headers={

        })
        res.encoding = 'gbk2312'
        soup = BeautifulSoup(res.text, 'html.parser')
        ele = soup.find('tbody')
        if ele is not None:
            for e in ele.find_all('tr'):
                record = list(e.find_all('td'))
                pinyin = record[0]
                words = record[1]
                all_words.append([tone+1, rhyme, pinyin.text, words.text])
        time.sleep(1)
    return all_words


if __name__ == '__main__':
    # all_rhymes = ['i2']
    # aw = []
    # for r in all_rhymes:
    #     aw += get_words(r)
    # expand = []
    # for i in aw:
    #     for word in i[3]:
    #         expand.append([i[0], i[1], i[2], word])
    # df = pd.DataFrame(expand, columns=['tone', 'rhyme', 'pinyin', 'word'])
    # df.to_csv('word_rhyme_new2.csv', index=False)
    df1 = pd.read_csv('word_rhyme_new.csv')
    df2 = pd.read_csv('word_rhyme.csv')
    df3 = pd.read_csv('word_rhyme_new2.csv')
    df = pd.concat([df1, df2, df3])
    df.to_csv('word_rhyme_con.csv', index=False)
