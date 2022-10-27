from models.model import Model


def main(rhythmic, title):
    my_model = Model.init_model()
    my_model(rhythmic, title)


if __name__ == '__main__':
    test = ['', '千古兴亡多少事']
    main(test[0], test[1])
