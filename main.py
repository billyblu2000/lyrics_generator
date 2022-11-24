from models.model import Model
import time


def main(rhythmic, title):
    my_model = Model.init_model()
    for i in range(3):
        s = time.time()
        my_model(rhythmic, title)
        print('time cost: ', time.time() - s)


if __name__ == '__main__':
    test = ['', '秋天']
    main(test[0], test[1])
