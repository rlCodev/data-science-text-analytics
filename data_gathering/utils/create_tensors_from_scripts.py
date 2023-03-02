import spacy
from tqdm import tqdm
import glob
import pandas as pd
import re
from sentence_transformers import SentenceTransformer

files = glob.glob('../../data/script/*', recursive=True)
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
    
#resultFile = open('C:/Users/Jakob/Documents/DSTA_Project/data-science-text-analytics/data_gathering/baseline/output/imdb_id_with_embSentencesList.txt', 'w')
imdb_id_with_tensor = []
for idx, filepath in tqdm(enumerate(files), total=len(files)):
    
    if(idx == 113):
        print("skipping")
        continue

    with open(filepath, 'r') as f:
        try:
            script = f.read()
        except:
            print("skipping")
            continue
    
    imdb_id = filepath.split('\\')[-1].split('.')[0]
    
    list_of_sentences = split_sentences(script)
    try:
        sentences_emb = get_sentence_emb(list_of_sentences)
    except:
        print("skipping")
        continue
    imdb_id_with_tensor.append([imdb_id, sentences_emb])

    if(idx % 50 == 0):
        df = pd.DataFrame(imdb_id_with_tensor, columns=['imdb_id', 'sentences_emb'])
        print("last entry in df = {0}".format(imdb_id))
        df.to_pickle('../baseline/output/imdb_id_with_embSentencesList{0}.pkl'.format(idx))


df = pd.DataFrame(imdb_id_with_tensor, columns=['imdb_id', 'sentences_emb'])
print(df)
df.to_pickle('../baseline/output/imdb_id_with_embSentencesList.pkl')