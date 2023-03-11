from models.model import Model, Baseline
import time

# model = Model.init_model()
model = Baseline.init_model()

def main(rhythmic, title):
    for i in range(3):
        s = time.time()
        model(rhythmic, title)
        print('time cost: ', time.time() - s)


if __name__ == '__main__':
    main('', '枯藤老树昏鸦')