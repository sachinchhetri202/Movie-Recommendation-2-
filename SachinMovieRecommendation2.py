import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from Levenshtein import distance as levenshtein_distance

def euclidean_distance(base_case_year, comparator_year):
    return abs(int(base_case_year) - int(comparator_year))

def jaccard_similarity(base_case_genres: str, comparator_genres: str):
    base_case_genres = set(base_case_genres.split(';'))  # cast list to a set
    comparator_genres = set(comparator_genres.split(';'))  # cast list to a set
    numerator = len(base_case_genres.intersection(comparator_genres))
    denominator = len(base_case_genres.union(comparator_genres))
    return float(numerator) / float(denominator)  # cast as float

def cosine_similarity_desc(base_case_desc, comparator_desc):
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform([base_case_desc, comparator_desc])
    similarity = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
    return similarity

df = pd.read_csv('movies.csv')

while True:
    searchTitle = input("\nInput a movie title to search: ").lower()
    print()
    df["result"] = df["title"].str.lower().str.find(searchTitle)
    print(df[['title']].loc[df["result"] == 0], '\n')
    if (df.loc[df["result"] == 0].empty):
        print("Movie not found. Search again....")
    else:
        searchID = input("Input the correct movie ID, or 0 if not on list: ")
        if (int(searchID) == 0):
            continue
        print()
        break

movie = df.loc[int(searchID)]

clustertype = int(input("Create cluster based on 1: title, 2: genre, or 3: both: "))

if clustertype == 1:
    cluster = df
if clustertype == 2:
    df["jaccard"] = df["genres"].map(lambda x: jaccard_similarity(movie['genres'], x))
    cluster = df.loc[df["jaccard"] > .65]
if clustertype == 3:
    cluster = df

K = int(input("How many results would you like?: "))

descWeight = 0
titleWeight = 0
yearWeight = 0

print("You may sort by description, title, and/or year.")
print("Would you like to sort by description? Y/N")
if (input().upper() == "Y"):
    print("What weight (0.0-1.0) would you like the description to have?")
    descWeight = float(input())

if (input("Would you like to sort by title? Y/N: ").upper() == "Y"):
    print("What weight (0.0-1.0) would you like the title to have?")
    titleWeight = float(input())

if 'year' in df.columns:
    print("Would you like to sort by year? Y/N")
    if (input().upper() == "Y"):
        print("What weight (0.0-1.0) would you like the year to have?")
        yearWeight = float(input())
        # Euclidean Year
        df['euclidean'] = df['year'].map(lambda x: euclidean_distance(movie['year'], x))
        df["euclidean"] = 1 - df["euclidean"] / 10 * yearWeight

if 'description' in df.columns and descWeight > 0:
    df['cosine'] = df['description'].map(lambda x: cosine_similarity_desc(movie['description'], x))

if 'title' in df.columns and titleWeight > 0:
    df['levenshtein'] = df['title'].map(lambda x: levenshtein_distance(movie['title'], x))

df["weight"] = descWeight * df.get('cosine', 0) + titleWeight * df.get('levenshtein', 0) + yearWeight * df.get(
    'euclidean', 0)
df = df.sort_values(by="weight", ascending=False)
print(df.head(K))
