import pandas as pd

def euclidean_distance(base_case_year, comparator_year):
    return abs(int(base_case_year) - int(comparator_year))

def jaccard_similarity(base_case_genres: str, comparator_genres: str):
    base_case_genres = set(base_case_genres.split(';')) # cast list to a set
    comparator_genres = set(comparator_genres.split(';')) # cast list to a set
    numerator = len(base_case_genres.intersection(comparator_genres))
    denomenator = len(base_case_genres.union(comparator_genres))
    return float(numerator) / float(denomenator) # cast as float

df = pd.read_csv('movies.csv')

while True:
    print('\n', "Input a movie title to search")
    searchTitle = input()
    print()
    df["result"] = df["title"].str.find(searchTitle)
    print(df[['title']].loc[df["result"] == 0], '\n')
    if (df.loc[df["result"] == 0].empty):
        print("Movie not found. Search again")    
    else:
        print("Input the correct movie ID, or 0 if not on list")
        searchID = input()
        if (int(searchID) == 0):
            continue
        print()
        break

movie = df.loc[int(searchID)]

print("Create cluster based on 1: title, 2: genre, or 3: both")
clustertype = int(input())

if clustertype == 1:
    cluster = df
if clustertype == 2:
    df["jaccard"] = df["genres"].map(lambda x : jaccard_similarity(movie['genres'], x))
    cluster = df.loc[df["jaccard"] > .65]
if clustertype == 3:
    cluster = df

#print(cluster)
    
print("How many results would you like?")
K = int(input())

descWeight = 0
titleWeight = 0
yearWeight = 0

print("You may sort by description, title, and/or year.")
print("Would you like to sort by description? Y/N")
if (input() == "Y"):
    print("What weight (0.0-1.0) would you like the description to have?")
    descWeight = input()
    df['cosine'] = descWeight

print("Would you like to sort by title? Y/N")
if (input() == "Y"):
    print("What weight (0.0-1.0) would you like the title to have?")
    titleWeight = input()
    df['levenshtein'] = titleWeight

print("Would you like to sort by year? Y/N")
if (input() == "Y"):
    print("What weight (0.0-1.0) would you like the ear to have?")
    yearWeight = input()
    #Euclidean Year
    df['euclidean'] = df['year'].map(lambda x: euclidean_distance(movie['year'], x))
    df["euclidean"] = 1 - df["euclidean"] / 10 * yearWeight

df["weight"] = df['cosine'] + df['levenshtein'] + df['Euclidean']
df.sort_values(df["weight"])
print(df.head(K))