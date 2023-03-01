from imdb import Cinemagoer
from tqdm import tqdm

# create an instance of the Cinemagoer class
ia = Cinemagoer()

with open('data_exploration/baseline/imdb_id_with_age_rating_list.txt', 'r') as file:
    # read a list of lines into data
    dataset = file.readlines()
    print(len(dataset))

labeled = []

for movie_data in tqdm(dataset):
    movie_id = movie_data.split(',')[0]
    movie_id = movie_id.replace('tt', '')
    # get the parents guide for the movie
    pg = ia.get_movie_parents_guide(movie_id)
    # get the advisory votes
    votes = pg["data"]["advisory votes"]
    # print the votes
    movie_data = movie_data.replace("\n", "")
    for vote in votes.items():
        print(vote)
        movie_data = movie_data.split(",")[0] + "," + movie_data.split(",")[1] +  "," + vote[0] + "," + str(vote[1]['votes']['None']) + "," + str(vote[1]['votes']['Mild']) + "," + str(vote[1]['votes']['Moderate']) + "," + str(vote[1]['votes']['Severe'])
        text = movie_data + "\n"
        labeled += text


# write the data to a file
with open('data_exploration/baseline/imdb_id_with_age_rating_and_labels.txt', 'a') as file:
    file.writelines(labeled)