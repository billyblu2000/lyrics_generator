from models.sentence_transformer_retriever import SentenceTransformersRetriever
import config


class Model:

    def __init__(self, config, retriever):
        if config.print_log:
            print('Init Model: Model...')
        self.config = config
        self.retriever = retriever

    @classmethod
    def init_model(cls):
        retriever = SentenceTransformersRetriever(config)
        return Model(config, retriever)

    def run(self, rhythmic: str, title: str):
        if config.print_log:
            print(f'Running model with rhythmic "{rhythmic}" and title "{title}"...')
        phrases = self.retriever(title)
        print(list(phrases['phrase']))

    def __call__(self, rhythmic: str, title: str):
        return self.run(rhythmic, title)
