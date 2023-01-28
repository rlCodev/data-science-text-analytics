import requests,json,csv,os

Movie_name = '8 mm'

API_key = '2f336ef7520a9304940fd71509ade61c'
#Use this method to return the tmdb-id (as a String) of a movie. It uses the defualt API_key which is
#set to my API_key. The input is a string of searchwords.

def get_tmdb_id_by_name(searchwords: str) -> str:
    #splits the string into each individual search word and creates query, by appending search words in query.
    words = searchwords.split(' ')
    query = 'https://api.themoviedb.org/3/search/movie/?api_key='+API_key+'&query='
    for i in range(len(words) - 1):
        query += words[i] + '+'
    query += words[len(words) - 1]

    #executes query, if response code is 200, then the query was successful
    response =  requests.get(query)
    if response.status_code==200: 
        array = response.json()
        #text contains the query as a json-file
        text = json.dumps(array)
        #print(text)
    else:
        print("No results for searchwords --> error")
        return ("No results for searchwords --> error")
    
    #dataset is json file converted into a list, containing the elements as either lists or maps
    dataset = json.loads(text)
    #extracts the movie id of the very first result
    # print(dataset)
    try:
        tmdb_id = dataset['results'][0]['id']
        return str(tmdb_id)
    except:
        return str(-1)
    # print(tmdb_id)

def get_tmdb_id_by_imdb_id(imdb_id:str) -> str:
    query = 'https://api.themoviedb.org/3/find/'+imdb_id+'?api_key='+API_key+'&external_source=imdb_id'
    response =  requests.get(query)
    if response.status_code==200:
        array = response.json()
        #text contains the query as a json-file
        text = json.dumps(array)
        #print(text)
    else:
        print("No results for searchwords --> error")
        return ("No results for searchwords --> error")
    dataset = json.loads(text)
    #extracts the movie id of the very first result
    # print(dataset)
    try:
        tmdb_id = dataset['movie_results'][0]['id']
        return str(tmdb_id)
    except:
        return str(-1)

#Uses tmdb id as input and returns the age certification as a String

def get_age_certfication_by_tmdb_id(tmdb_id: str) -> str:
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
        # print(dataset)
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
    resultFile = open('.//imdb_id_with_age_rating_list.txt', 'w')
    inputFile = open('movie1K_list.txt')

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





#write a function to compose the query using the parameters provided
# def get_data(API_key, Movie_ID):
#     query = 'https://api.themoviedb.org/3/movie/'+Movie_ID+'?api_key='+API_key+'&language=en-US'
#     response =  requests.get(query)
#     if response.status_code==200: 
#     #status code == 200 indicates the API query was successful
#         array = response.json()
#         text = json.dumps(array)
#         return (text)
#     else:
#         return ("error")


# def get_imdb_id_by_tmdb_id(API_key, tmdb_id):
#     text = get_data(API_key,str(tmdb_id))
#     dataset = json.loads(text)
#     result = dataset['imdb_id']
#     return result

    
# text = get_data(API_key,str(id))
# dataset = json.loads(text)
# print('Data =' , dataset)
# result = dataset['imdb_id']
# print('\nResult =',result)




# def write_file(filename, text):
#     dataset = json.loads(text)
#     csvFile = open(filename,'a')
#     csvwriter = csv.writer(csvFile)
#     #unpack the result to access the "collection name" element
#     try:
#         collection_name = dataset['belongs_to_collection']['name']
#     except:
#         #for movies that don't belong to a collection, assign null
#         collection_name = None
#     result = [dataset['original_title'],collection_name]
#     # write data
#     csvwriter.writerow(result)
#     print (result)
#     csvFile.close()
#movie_list = ['464052','508442']
#write header to the file
#csvFile = open('movie_collection_data.csv','a')
#csvwriter = csv.writer(csvFile)
#csvwriter.writerow(['Movie_name','Collection_name'])
#csvFile.close()


#for movie in movie_list:
    #text = get_data(API_key, movie)
    #make sure your process breaks when the pull was not successful 
    #it's easier to debug this way
    #if text == "error":
    #    break
    #write_file('movie_collection_data.csv', text)