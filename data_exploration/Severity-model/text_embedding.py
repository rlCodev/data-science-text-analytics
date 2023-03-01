import os
import argparse
import glob
import pandas as pd

from sentence_transformers import SentenceTransformer

parser = argparse.ArgumentParser()

# path to the train/dev/test sets.
parser.add_argument('--data_dir', type=str, default='./splited_data/')

# dataframes in .pkl files
list_of_filename = glob.glob(args.base_dir + "*.pkl")

### it is also possible to specify filenames by the user
# list_of_filename =[
#  './splited_data/violence_train.pkl',
#  './splited_data/violence_dev.pkl',
#  './splited_data/violence_test.pkl'
# ]

model = SentenceTransformer('stsb-bert-base', device='cuda:6')
#Sentences are encoded by calling model.encode()
def get_sentence_emb(whole_list_of_sentences):
    embeddings = model.encode(whole_list_of_sentences, convert_to_tensor = True)
    return embeddings

for each_file in list_of_filename:
    print("Working on: ", each_file)
    # read one file
    all_data = pd.read_pickle(each_file)
    # add sent emb column
    all_data['text_emb'] = all_data['text'].apply(get_sentence_emb)
    # remove textual data - optional
    all_data.drop(columns=['text'])    
    # save df, exclude ".pkl"
    all_data.to_pickle(each_file[:-4]+"_emb.pkl")
#     print(all_data)
    print(each_file[:-4]+"_emb.pkl -- finished.")