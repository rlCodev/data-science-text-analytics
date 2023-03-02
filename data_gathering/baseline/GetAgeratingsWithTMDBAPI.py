import requests,json,csv,os

API_key = '2f336ef7520a9304940fd71509ade61c'

def get_tmdb_id_by_name(searchwords: str) -> str:
    """
    Use this method to return the tmdb-id (as a String) of a movie. It uses the defualt API_key which is
    set to a personal API key. The input is a string of searchwords. If a movie is found, the tmdb-id is returned.
    """
    
    words = searchwords.split(' ') #splits the string into each individual search word and creates query, by appending search words in query.
    query = 'https://api.themoviedb.org/3/search/movie/?api_key='+API_key+'&query='
    for i in range(len(words) - 1):
        query += words[i] + '+'
    query += words[len(words) - 1]

    response =  requests.get(query) #executes query, if response code is 200, then the query was successful
    if response.status_code==200: 
        array = response.json()
        text = json.dumps(array) #text contains the query as a json-file
    else:
        print("No results for searchwords --> error")
        return ("No results for searchwords --> error")
    
    dataset = json.loads(text) #dataset is json file converted into a list, containing the elements as either lists or maps
    try:
        tmdb_id = dataset['results'][0]['id'] #extracts the movie id of the very first result
        return str(tmdb_id)
    except:
        return str(-1)

def get_tmdb_id_by_imdb_id(imdb_id:str) -> str:
    """
    Use this method to return the tmdb-id (as a String) of a movie, given the imdb-id.
    """

    query = 'https://api.themoviedb.org/3/find/'+imdb_id+'?api_key='+API_key+'&external_source=imdb_id'
    response =  requests.get(query)
    if response.status_code==200:
        array = response.json()
        text = json.dumps(array) #text contains the query as a json-file
    else:
        print("No results for searchwords --> error")
        return ("No results for searchwords --> error")
    
    dataset = json.loads(text)
    try:
        tmdb_id = dataset['movie_results'][0]['id'] #extracts the movie id of the very first result
        return str(tmdb_id)
    except:
        return str(-1)

#Uses tmdb id as input and returns the age certification as a String

def get_age_certfication_by_tmdb_id(tmdb_id: str) -> str:
    """
    This is a crucial part of our data preparation. We need to know the age rating of a movie,
    in order to set this as a label for our classifier. This method uses the tmdb-id as input and
    returns the age rating as a String. If no age rating is found, the method returns 'no age rating found'.
    """

    if(tmdb_id == '-1'):
        return 'no age rating found'
    else:
        query = 'https://api.themoviedb.org/3/movie/'+tmdb_id+'/release_dates?api_key='+API_key
        response =  requests.get(query)
        array = response.json()
        text = json.dumps(array)
        dataset = json.loads(text)

        list_of_certs = dataset['results']
        index = -1
        for i in range(len(list_of_certs)):
            if(list_of_certs[i]['iso_3166_1'] == 'US'):
                index = i
                break

        if(index != -1):
            age_rating = dataset['results'][index]['release_dates'][0]['certification']
        else:
            return 'no age rating found'
    
    if(age_rating == ''):
        return 'no age rating found'
    return age_rating


def main():
    print(os.path.dirname(__file__))
    resultFile = open('./output/imdb_id_with_age_rating_list.txt', 'w')
    inputFile = open('./input/movie1K_list.txt')

    output = {}
    
    for line in inputFile:
        imdb_id = line.strip()
        id = get_tmdb_id_by_imdb_id(imdb_id)
        age_rating = get_age_certfication_by_tmdb_id(id)
        output.update({imdb_id : age_rating})
        resultFile.write(imdb_id + ',' + age_rating + '\n')
        print('ID = ' + imdb_id + ', age rating = ' + age_rating)

def test():
    id = get_tmdb_id_by_imdb_id("tt2258281")
    age_rating = get_age_certfication_by_tmdb_id(id)
    print(age_rating)


if __name__ == "__main__":
    main()