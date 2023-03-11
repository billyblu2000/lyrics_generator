import os.path

# project
print_log = True

# data path
project_path = os.path.dirname(os.path.abspath(__file__))
data_folder = os.path.join(project_path, 'data')
phrase_database = os.path.join(data_folder, 'phrase_database_d0.csv')
rhythmic_index = os.path.join(data_folder, 'rhythmic_index.csv')
author_index = os.path.join(data_folder, 'author_index.csv')
phrase_embedding = os.path.join(data_folder, 'phrase_embs_sentence_granu.npz')
sentence_transformer_path = 'sentence-transformers/distiluse-base-multilingual-cased'
npp_model_checkpoint_path = os.path.join(data_folder, 'model_bert_nsp_pretrain_4class.bin')
# npp_model_checkpoint_path = None
rhyme_lookup_table = os.path.join(data_folder, 'word_rhyme_con.csv')
baseline_sentence_database = os.path.join(data_folder, 'all_sentence.json')
baseline_sentence_embedding = os.path.join(data_folder, 'sentence_embs.npz')

# hyper parameters
retrieve_num = 300
