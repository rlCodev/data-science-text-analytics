import requests,json,csv,os

Movie_name = '8 mm'
titles = ['A-Night-at-the-Roxbury',
 'Manhunte',
 'Life-Aqua',
 'In-The-Bedroom',
 'Romy-and-Michelles-High-School-Reunion',
 'Notting-Hill',
 'Agnes-of-God',
 'Exo',
 'Romeo-Julie',
 'Alien-Nation',
 'Murderland',
 'Commando',
 'The-Forsaken',
 'Jerry-Maguire',
 'The-Jerk',
 '2010',
 '10-Things-I-Hate-About-You',
 'Going-for-the-Gold',
 'Graduate-The',
 'Father-of-the-Bride',
 'Arctic-Blue',
 'Ghostbuste',
 'Battle-of-Algie',
 'Mean-Stree',
 'The-Godfather-Part-III',
 'Halloween',
 'Field-of-Dream',
 'Jackie-Brown',
 'Stepmom',
 'Schindlers-L',
 'The-God-Fathe',
 'Bamboozeled',
 'At-First-Site',
 'Duel',
 'Shakespeare-in-Love',
 'Broadcast-New',
 'The-Chronicles-of-Narnia-The-Lion-the-Witch-and-the-Wardrobe',
 'Million-Dollar-Baby',
 'Trainspotting',
 'Big',
 'Bridges-of-Madison-County-The',
 'Walking-Tall',
 'Fear-and-Loathing-In-Las-Vega',
 'American-Beauty',
 'What-Women-Wan',
 'Breakfast-Club',
 'Point-Break',
 'The-Constant-Gardene',
 'Annie-Hall',
 'part-1',
 'The-Thing',
 'Pretty-Woman',
 'Tomorrow-Never-Die',
 'Ghostbusters-II',
 'Gladiato',
 'Heavenly-Creature',
 'Halloween-H20',
 'Treasure-of-the-Sierra-Madre',
 'Braveheart-S',
 '1492The-Conquest-of-Paradise',
 'Mr-Brook',
 'American-Graff',
 'Ren',
 'Millers-Crossing',
 'My-Own-Private-Idaho',
 'Dave-Barrys-Complete-Guide-to-Guy',
 'Bean',
 'Bull-Durham',
 'Zerophilia',
 'RKO-281',
 'Assassin',
 'True-Romance',
 'Last-Action-Here-shooting-draf',
 'Poetic-Justice',
 'Meet-Joe-Black',
 'Working-Girl',
 'The-Produce',
 'McCabe-and-Mrs-Mille',
 'Ten-Things-I-Hate-About-You',
 'Arcade',
 '48-Hou',
 'Glengarry-Glen-Ro',
 'The-Surfer-King',
 'Traff',
 'Chillfacto',
 'Bound-First-Draf',
 'Zulu-Dawn',
 'Clue',
 'Truman-Show-The',
 'Asylum',
 'Hacke',
 'The-Godfather-II',
 'Quiz-Show',
 'Gandh',
 'Basquia',
 'World-Is-Not-Enough',
 'Come-See-the-Paradise',
 'Streetwise',
 'Few-Good-Men-A',
 'Strange-Brew',
 'Suburbia',
 'Emma',
 'Deep-End-of-the-Ocean',
 'Sixth-Sense-The',
 'AirForce-One',
 'The-World-is-Not-Enough',
 'Wild-Hog',
 'Last-of-the-Mochican',
 'Messenge',
 'Titan',
 'Wonderland',
 'Sweet-Smell-of-Suce',
 'Pirscilla-Queen-of-the-Dese',
 'The-Ringe',
 'Indiana-Jones-IV',
 'Girl-with-a-Pearl-Earring',
 'Superman-Live',
 'Napoleon-Dynamite',
 'Malcom-X',
 'Avenge',
 'MASH',
 'Last-Action-Hero-first-draf',
 'Ronin',
 'Chinatown',
 'The-French-Connection',
 'The-Sting',
 'Pump-Up-The-Volume',
 'Hannah-and-her-Siste',
 'Three-Days-of-the-Condo',
 'True-Lie',
 'My-Best-Friends-Wedding',
 'Braveheart-Tran',
 'Kid',
 'Clerk',
 'One-Saliva-Bubble',
 'The-Elephant-Man',
 'Theres-Something-about-Mary',
 'Fantasia-2000',
 'Sleepless-in-Seattle',
 'The-Doom-Generation',
 'Office-Space',
 'Stir-Of-Echoe',
 'Roboco',
 'Big-Lebowsk',
 'Alien',
 'In-The-Heat-of-the-Nigh',
 'Rocky',
 'Primary-Evidence',
 'Barton-Fink',
 'Body-of-Evidence',
 'Airplane',
 'This-Is-Spinal-Ta',
 'Aliens-Vs-Predato',
 'Apocalypse-Now',
 'Memoirs-of-a-Geisha',
 'Independence-Day',
 'The-Apartmen',
 'Taxi-Drive',
 'Man-on-the-Moon',
 'one',
 'Crying-Game',
 'Manhun',
 'Kramer-versus-Krame',
 'As-Good-As-It-Ge',
 'Only-You',
 '8-MM',
 '9th-Gate',
 'Horse-Whisperer-The',
 'Space-Ball',
 'Do-the-Right-Thing',
 'Smoke',
 'Bound',
 'Mission-Impossible-2',
 'Alien-3',
 'Mr-and-Mrs-Smith',
 'Boys-on-the-Side',
 'Red-White-Black-and-Blue',
 'Rock-The',
 'Runaway-Bride',
 'Dog-Day-Afternoon',
 'Crash',
 'Karate-Kid',
 'Crouching-Tiger-Hidden-Dragon',
 'Soldie',
 'Schindlers-List',
 'Rocky-Horror-Picture-Show-The',
 'Breakfast-Club-The',
 'Copycat',
 'Titanic',
 'Notting-Hill',
 'Platoon',
 'Total-Recall',
 'As-Good-As-It-Gets',
 'Fargo',
 'Youve-got-Mail',
 'Twin-Peaks',
 'Labyrinth',
 'Fight-Club',
 'Good-Will-Hunting',
 'Jerry-Maguire',
 'Kill-Bill',
 '10-Things-I-Hate-About-You',
 'Willow',
 'Graduate-The',
 'Witness',
 'Jackie-Brown',
 'Stepmom',
 'Lethal-Weapon',
 'Crow-The',
 'LA-Confidential',
 'Shakespeare-in-Love',
 'Highlander',
 'Top-Gun',
 'Amadeus',
 'Trainspotting',
 'Bridges-of-Madison-County-The',
 'Crowded-Room-A',
 'Lost-in-Space',
 'Star-Wars-VI-Return-of-the-Jedi',
 'American-Beauty',
 'Highlander-III',
 'Terminator-2',
 'Sphere',
 'Pretty-Woman',
 'Scream',
 'Casino',
 'Scream-2',
 'Natural-Born-Killers',
 'Dr-Strangelove',
 'Escape-from-LA',
 'Hackers',
 '48-Hours',
 'Purple-Rain',
 'Dead-Poets-Society',
 'Indiana-Jones-Temple-of-Doom',
 'Ghostbusters',
 'True-Romance',
 'Gone-in-60-Seconds',
 'Meet-Joe-Black',
 'Broadcast-News',
 'White-Squall',
 'Men-In-Black',
 'Starship-Troopers',
 'Die-Hard',
 'Tron',
 'Indiana-Jones-Last-Crusade',
 'Seven',
 'Truman-Show-The',
 'From-Dusk-Til-Dawn',
 'Dune',
 'Troops',
 'Strange-Days',
 'American-President-The',
 'Reservoir-Dogs',
 'Basic-Instinct',
 'Get-Shorty',
 '2001-A-Space-Odyssey',
 'Sixth-Sense-The',
 'Matrix-The-(1996)',
 'Interview-with-the-Vampire',
 'Usual-Suspects-The',
 'True-Lies',
 'PredatorHunter',
 'Star-Wars-IV-A-New-Hope',
 'Matrix-The-(1997)',
 'Assassins',
 'Big-Lebowski-The',
 'Abyss-The',
 'Ronin',
 'Legend-of-Darkness',
 'Silence-of-the-Lambs-The',
 'Escape-from-New-York',
 'Kramer-vs-Kramer',
 'My-best-Friends-Wedding',
 'Matrix-Reloaded-The',
 'Green-Mile-The',
 'Sleepless-in-Seattle',
 'Some-Like-It-Hot',
 'Wizard-of-Oz-The',
 'Alien',
 'Saving-Private-Ryan',
 'Friday-the-13th',
 'Body-of-Evidence',
 'Pulp-Fiction',
 'Batman-and-Robin',
 'Apocalypse-Now',
 'Blade-Runner',
 'Independence-Day',
 'White-Angel',
 'Jurassic-Park',
 'Tomorrow-Never-Dies',
 'Terminator-The',
 'Nightmare-on-Elmstreet',
 'Wild-at-Heart',
 'Horse-Whisperer-The',
 'World-is-not-Enough-The',
 'Ferris-Buellers-Day-Off',
 'Alien-3',
 'Rock-The',
 'Runaway-Bride',
 'Four-Rooms',
 'Big-Eyes',
 'Captain-Fantastic',
 'Place-Beyond-the-Pine-The',
 'Judas-and-the-Black-Messiah',
 'Saving-Mr-Banks',
 'On-the-Basis-of-Sex',
 'If-Beale-Street-Could-Talk',
 'Slow-West',
 'Omen-The',
 'Borat-Subsequent-Moviefilm',
 'Boxtrolls-The',
 'Star-Wars-Episode-III-Revenge-of-the-Sith',
 'Arrival',
 'Avengers-Endgame',
 'Blade-Runner-2049',
 'Minari',
 'Great-Gatsby-The',
 'Notting-Hill',
 'Man-Who-Invented-Christmas-The',
 'Shrek',
 'Black-Panther',
 'Scandal-Final-Pilot-TV-Script-PDF',
 'Beauty-and-the-Beast',
 'Animal-Kingdom',
 'Eternal-Sunshine-of-the-Spotless-Mind',
 'Trumbo',
 'Philomena',
 'Brads-Status',
 'Woman-in-Gold',
 'Knives-Out',
 'Incredibles-The',
 'Martian-The',
 'Guardians-of-the-Galaxy-2',
 'War-for-the-Planet-of-the-Apes',
 'Masters-of-Horror',
 'Casablanca',
 'Good-Will-Hunting',
 'The-Irishman',
 'Artist-The',
 'Greys-Anatomy-Episode-201-TV-Script-PDF',
 'Bridget-Joness-Baby',
 'Mank',
 'Midnight-in-Paris',
 'Room-The',
 'Big-Short-The',
 'Outlander-Sassenach',
 'Mud',
 'Flight',
 'Whiplash',
 'Her-Smell',
 'El-Mariachi',
 'Elizabeth-Blue',
 'Victoria-and-Abdul',
 'Sound-of-Metal',
 'Despicable-Me-2',
 'Good-Wife-The-Stripped',
 'Downsizing',
 'Roma',
 'Still-Alice',
 'WALL-E',
 'Ted-Lasso',
 'Jackie',
 '1917',
 'Into-the-Woods',
 'Straight-Outta-Compton',
 'Pain-and-Glory',
 'Good-Wife-The',
 'Uncut-Gems',
 'Skyfall',
 'Wreck-it-Ralph',
 'Lego-Movie-The',
 'Grand-Budapest-Hotel-The',
 'Fleabag',
 'White-Tiger-The',
 'Booksmart',
 'Wadjda',
 'Inglourious-Basterds',
 'Before-Midnight',
 'Greys-Anatomy-Episode-202-TV-Script-PDF',
 'Star-Is-Born-A',
 'Love-is-Strange',
 'How-to-Train-Your-Dragon-2',
 'Inside-Out',
 'First-Man',
 'Promising-Young-Woman',
 'Friends-Episode-108-The-One-Where-Nana-Dies-Twice',
 'Can-You-Ever-Forgive-Me',
 'Foxcatcher',
 'Get-Low',
 'Wolf-of-Wall-Street-The',
 'Crazy-Rich-Asians',
 'Queens-Gambit-The',
 'Midsommar',
 'Kings-Speech-The',
 'Frankenweenie',
 'Just-Mercy',
 'Hail-Caesar',
 'Lobster-The',
 'Help-The',
 'Pose',
 'Foxtrot',
 'Moonlight',
 'Florida-Project-The',
 'Parasite',
 'Brooklyn',
 'American-Hustle',
 'Last-Ship-The',
 'Legally-Blonde',
 'Disaster-Artist-The',
 'Cars-2',
 'Steve-Jobs',
 'Invisible-Woman-The',
 'Lion-King-The',
 'Queen-and-Slim',
 'Soul',
 'Social-Network-The',
 'Juno',
 'Gravity',
 'Jane-Eyre',
 'Trial-of-the-Chicago-7-The',
 'How-to-Get-Away-with-Murder-Pilot-TV-Script-PDF',
 'Argo',
 'Ill-See-You-in-My-Dreams',
 'Despicable-Me',
 'Lion',
 '12-Years-a-Slave',
 'Paranorman',
 'Legend',
 'Leisure-Seeker',
 'Man-Up',
 'Bird-Box',
 'Half-of-It-The',
 'Gone-Girl',
 'Green-Lantern',
 'Three-Billboards-Outside-Ebbing-Missouri',
 'War-Horse',
 'Father-The',
 'Hunger-Games-The',
 'Hateful-Eight-The',
 'This-is-40',
 'Learning-to-Drive',
 'Harry-Potter-and-the-Half-Blood-Prince',
 'Ex-Machina',
 'Lost-City-of-Z-The',
 'Anchorman',
 'I-Tonya',
 'Vice',
 'Thor-Ragnarok',
 'Im-Thinking-of-Ending-Things',
 'Avengers-The',
 'Farewell-The',
 '42',
 'Kids-Are-Alright-The',
 'American-Sniper',
 'This-Is-Us',
 'First-Reformed',
 'Call-Me-By-Your-Name',
 'Wonderstruck',
 'Kubo-and-the-Two-Strings',
 'While-Were-Young',
 'Mary-Queen-of-Scots',
 'Minions',
 'Wonder-Woman',
 'Hell-or-High-Water',
 'Lincoln',
 'Ford-v-Ferrari',
 'Ted',
 'Lady-Bird',
 'Little-Women',
 'Prom-The',
 'Mollys-Game',
 'Fault-in-Our-Stars-The',
 'Brigsby-Bear',
 'Concussion',
 'Unbroken',
 'Scandal-Episode-301-TV-Script-PDF',
 'Trainwreck',
 'Big-Bang-Theory-The',
 'Farmageddon',
 'Greys-Anatomy-Early-Pilot-TV-Script-PDF',
 'Tower-Heist',
 'Imitation-Game',
 'Battle-of-the-Sexes',
 'Bourne-Identity-The',
 'Spotlight',
 'Greys-Anatomy-Final-Pilot-TV-Script-PDF',
 'Da-5-Bloods',
 'Breathe',
 'Theory-of-Everything-The',
 'Mother-and-Child',
 'Crown-The',
 '101-Dalmatians',
 'Stranger-Things',
 'Birdman',
 'Boyhood',
 'My-Big-Fat-Greek-Wedding-2',
 'Rust-and-Bone',
 'Zootopia',
 'Big-Bang-Theory-The-The-Griffin-Equivalency',
 'Tenet',
 'Colette',
 'Scandal-Early-Pilot-1-TV-Script-PDF',
 'Django-Unchained',
 'Star-Trek',
 'Gold',
 'How-to-Train-Your-Dragon-The-Hidden-World',
 'I-Smile-Back',
 'Interstellar',
 'Mr-Turner',
 'BlacKkKlansman',
 'Manchester-by-the-Sea',
 'Pawn-Sacrifice',
 'DeepStar-Six',
 'Beasts-of-the-Southern-Wild',
 'Game-of-Thrones',
 'Deadpool',
 'Darkest-Hour',
 'Eye-in-the-Sky',
 'Mudbound',
 'Danish-Girl-The',
 'United-States-vs-Billie-Holiday-The',
 'Marriage-Story',
 'Fantastic-Woman-A',
 'Iron-Lady',
 'Dallas-Buyers-Club',
 'Ballad-of-Buster-Scruggs-The',
 '20th-Century-Women',
 'Greys-Anatomy-Episode-108-TV-Script-PDF',
 'Rush',
 'Harry-Potter-and-the-Chamber-of-Secrets',
 'Creed',
 'First-They-Killed-My-Father',
 'Joker',
 'Fences',
 'One-Night-in-Miami',
 'Win-Win',
 'Tinker-Tailor-Solider-Spy',
 'La-La-Land',
 'Scandal-Early-Pilot-2-TV-Script-PDF',
 'Bridge-of-Spies',
 'Descendants',
 'Alien',
 'Past-The',
 'Croods-The',
 'Girl-with-the-Dragon-Tattoo-The',
 'Pulp-Fiction',
 'Nebraska',
 'Logan',
 'Mississippi-Grind',
 'Kill-Your-Darlings',
 'Life-of-Pi',
 'Big-Sick-The',
 'Beguiled-The',
 'Ma-Raineys-Black-Bottom',
 'Two-Popes-The',
 '22-July',
 'Jojo-Rabbit',
 'Coco',
 'Mad-Men',
 'Dunkirk',
 'Dear-White-People',
 'Her',
 'Macbeth',
 'Honey-Boy',
 'Good-Place-The',
 'What-They-Had',
 'Favourite-The',
 'Nomadland',
 'Aeronauts-The',
 'Wild',
 'Spider-Man-Into-the-Spider-Verse',
 'Hidden-Figures',
 'Shape-of-Water-The',
 'Get-Out',
 'Girl-on-a-Train-The',
 'Frozen',
 'Silver-Linings-Playbook',
 'Green-Book']

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
        print(list_of_certs)
        index = -1
        # print(dataset)
        for i in range(len(list_of_certs)):
            if(list_of_certs[i]['iso_3166_1'] == 'US'):
                index = i
                break
        try:
            age_rating = dataset['results'][index]['release_dates'][0]['certification']
        except:
            return 'no age rating found'
    
    if(age_rating == ''):
        return 'no age rating found'
    return age_rating


def main():
    #f = open('.//map_title_to_ageRating.txt', 'w')
    #output = {}
    
    #for title in titles:
    #    id = get_tmdb_id_by_name(title)
    #    age_rating = get_age_certfication_by_tmdb_id(id)
    #    output.update({title : age_rating})
    #    f.write(title + ',' + age_rating + '\n')
    #    print('title = ' + title + ', age rating = ' + age_rating)
    id = get_tmdb_id_by_imdb_id("tt0468569")
    print(id)
    age_rating = get_age_certfication_by_tmdb_id(id)
    print('age rating = ' + age_rating)


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