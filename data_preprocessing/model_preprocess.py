import pandas as pd 
import numpy as np
import pickle
print(pd.__version__)

def get_script_from_id(id):
    script = open('../data/script/' + id + '.script', 'r').read()
    # print(script)
    script = script.replace("'", " ").replace('"', ' ').replace('\n', ' ').replace('\r', ' ').replace('\t', ' ').replace('\b', ' ').replace('\\', ' ')
    return script

inputFile = open('C:\\Users\\Jakob\\Documents\\DSTA_Project\\data-science-text-analytics\\data_gathering\\baseline\\output\\imdb_id_with_age_rating_and_labels.txt')
df_data = []
for line in inputFile:
    line_data = line.strip().split(',')
    # print(line_data)
    line_data.append(int(line_data[3]) + int(line_data[4]) + int(line_data[5]) + int(line_data[6]))
    
    max_index = 0
    max_value = 0
    for i in range(3,7):
        vote_count = int(line_data[i])
        if(vote_count >= max_value):
            max_index = i - 3
            max_value = vote_count
    line_data.append(max_index)
    try:
        script = get_script_from_id(line_data[0])
    except:
        #print('Error on loading script for id: ' + line_data[0])
        continue
    line_data.append(script)
    df_data.append(line_data)

# id | Aspect | None | Mild | Moderate | Severe | Total_votes | Aspect_rating | text
df = pd.DataFrame(df_data, columns=['id', 'age_rating', 'Aspect', 'None', 'Mild', 'Moderate', 'Severe', 'Total_votes', 'Aspect_rating', 'text'])
df.drop(columns=["age_rating"], inplace=True)
df.reset_index(drop=True, inplace=True)
df = df.astype({'Mild':'int', 'Moderate':'int', 'Severe':'int', 'None':'int', 'Total_votes':'int', 'Aspect_rating':'int'})
df.head()

def map_data_frame_to_pickle(data_frame, file_name, path='./'):
    file_name = f'{file_name}.pkl'
    with open(path + file_name, 'wb') as f:
        data_frame.to_pickle(f, protocol=4)

def split_df_into_test_and_train(aspect_name, df, path='./'):
    df_train = df.sample(frac=0.85, random_state=0)
    df_test = df.drop(df_train.index)
    map_data_frame_to_pickle(df_train, f'{aspect_name}_train', 'C:\\Users\\Jakob\\Documents\\DSTA_Project\\data-science-text-analytics\\data\\pickle')
    map_data_frame_to_pickle(df_test, f'{aspect_name}_test', 'C:\\Users\\Jakob\\Documents\\DSTA_Project\\data-science-text-analytics\\data\\pickle')
    map_data_frame_to_pickle(df_test, f'{aspect_name}_dev', 'C:\\Users\\Jakob\\Documents\\DSTA_Project\\data-science-text-analytics\\data\\pickle')

# group the dataframe by 'aspect' and create a dictionary of dataframes
df_dict = {aspect: aspect_df.drop('Aspect', axis=1) for aspect, aspect_df in df.groupby('Aspect')}

# print the dictionary of dataframes
for aspect, aspect_df in df_dict.items():
    print(aspect, len(aspect_df))
    split_df_into_test_and_train(aspect, aspect_df, '../data/pickle/')
