import os.path

# project
print_log = True

# data path
data_folder = 'data'
phrase_database = os.path.join(data_folder, 'phrase_database_d0_drop_dup.csv')
rhythmic_index = os.path.join(data_folder, 'rhythmic_index.csv')
author_index = os.path.join(data_folder, 'author_index.csv')
phrase_embedding = os.path.join(data_folder, 'phrase_embs_sentence_granu.npz')
sentence_transformer_path = 'sentence-transformers/distiluse-base-multilingual-cased'

# hyper parameters
retrieve_num = 500
