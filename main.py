from models.model import Model
import time

my_model = Model.init_model()


def main(rhythmic, title):
    for i in range(3):
        s = time.time()
        my_model(rhythmic, title)
        print('time cost: ', time.time() - s)


if __name__ == '__main__':
    test = ['', '千古兴亡多少事']
    main(test[0], test[1])
    test = ['', '万里悲秋常作客']
    main(test[0], test[1])
    test = ['', '明月几时有']
    main(test[0], test[1])
    test = ['', '无可奈何花落去']
    main(test[0], test[1])
    test = ['', '枯藤老树昏鸦']
    main(test[0], test[1])
    test = ['', '对酒当歌']
    main(test[0], test[1])
