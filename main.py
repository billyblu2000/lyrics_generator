from models.model import Model
import time

def main(rhythmic, title):
    my_model = Model.init_model()
    my_model(rhythmic, title)


if __name__ == '__main__':
    test = [['', '千古兴亡多少事'],
            ]
    for i in test:
        s = time.time()
        main(i[0], i[1])
        print('time cost: ', time.time() - s)