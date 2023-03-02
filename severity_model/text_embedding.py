import os
import argparse
import glob
import pandas as pd
import spacy
import re

from sentence_transformers import SentenceTransformer

parser = argparse.ArgumentParser()

# path to the train/dev/test sets.
parser.add_argument('--data_dir', type=str, default='./splited_data/')
args = parser.parse_args()
# dataframes in .pkl files
list_of_filename = glob.glob(args.data_dir + "/*.pkl")
print(list_of_filename)


### it is also possible to specify filenames by the user
# list_of_filename =[
#  './splited_data/violence_train.pkl',
#  './splited_data/violence_dev.pkl',
#  './splited_data/violence_test.pkl'
# ]

model = SentenceTransformer('stsb-bert-base')
#Sentences are encoded by calling model.encode()
def get_sentence_emb(whole_list_of_sentences):
    embeddings = model.encode(whole_list_of_sentences, convert_to_tensor = True)
    #print(embeddings)
    return embeddings

def split_sentences(text):
    # define the pattern for splitting sentences using regex
    pattern = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s'
    # split the text into sentences using the pattern
    sentences = re.split(pattern, text)
    sentences = [re.sub(r'\s+', ' ', sentence).strip() for sentence in sentences]
    sentences = [sentence for sentence in sentences if sentence]
    return sentences

for each_file in list_of_filename:
    print("Working on: ", each_file)
    # read one file
    all_data = pd.read_pickle(each_file)
    print(all_data)
    # add sent emb column
    print(all_data['text'])
    sentences = all_data['text'].apply(split_sentences)
    all_lists_sentences = sentences.tolist()
    embedded_sentences = []
    for sent_list in all_lists_sentences:
        print(sent_list)
        get_sentence_emb(sent_list)
        embedded_sentences.append(get_sentence_emb(sent_list))
        print(embedded_sentences)
    all_data['text_emb'] = embedded_sentences #sentences.apply(get_sentence_emb)
    # remove textual data - optional
    all_data.drop(columns=['text'], inplace=True)    
    print(all_data)
    # save df, exclude ".pkl"
    all_data.to_pickle(each_file[:-4]+"_emb.pkl")
    #print(all_data)
    print(each_file[:-4]+"_emb.pkl -- finished.")

# CMD call:
# python text_embedding.py --data_dir C:\Users\Jakob\Documents\DSTA_Project\data-science-text-analytics\data\pickle