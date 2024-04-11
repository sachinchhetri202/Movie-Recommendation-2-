'''
Recommendation engine,  Part 2
-- Ease of use of GUI or text-based system.
-- K-means clustering is used with genres and title.
-- Cosine similarity on the description of the movies is used.
-- Levenshtein distance for the titles is used.
-- Euclidean distance with year is used.
'''

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from Levenshtein import distance as levenshtein_distance

# Configure pandas to display all columns and rows as needed, for better data visibility
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


# Define a function to calculate Jaccard similarity between two sets of genres
def jaccard_similarity(base_case_genres: str, comparator_genres: str):
    base_case_genres = set(base_case_genres.split(';'))
    comparator_genres = set(comparator_genres.split(';'))
    numerator = len(base_case_genres.intersection(comparator_genres))
    denominator = len(base_case_genres.union(comparator_genres))
    return numerator / denominator

# Define a function to calculate cosine similarity between descriptions of two movies
def cosine_similarity_desc(base_case_desc, comparator_desc):
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform([base_case_desc, comparator_desc])
    similarity = cosine_similarity(tfidf_matrix)[0][1]
    return similarity

# Define a function to calculate the Euclidean distance between the release years of two movies
def euclidean_distance(base_case_year, comparator_year):
    return abs(int(base_case_year) - int(comparator_year))

# Define a function for clustering movies based on specified attributes using KMeans
def kmeans_cluster(df, cluster_on, k=3):
    if cluster_on == 'genre':
        data_to_cluster = df['genres'].values.astype('U')
    elif cluster_on == 'title':
        data_to_cluster = df['title'].values.astype('U')
    elif cluster_on == 'both':
        data_to_cluster = (df['title'] + " " + df['genres']).values.astype('U')
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(data_to_cluster)
    kmeans = KMeans(n_clusters=k, random_state=42).fit(tfidf_matrix)
    df['cluster'] = kmeans.labels_
    return df

# Load the movie dataset
df = pd.read_csv('movies.csv')

# Loop to allow user to search and select a movie by title
while True:
    search_title = input("\nInput a movie title to search: ").lower()
    print()
    search_results = df[df['title'].str.lower().str.contains(search_title)]
    if search_results.empty:
        print("Movie not found. Please try again.")
    else:
        print(search_results[['title']], '\n')
        search_id = input("Input the correct movie ID to select, press Enter to continue searching, or input any key followed by Enter to proceed: ")
        if search_id.isdigit():
            print()
            break
        elif search_id == "":
            continue
        else:
            print("Invalid input. Please try again or just press Enter to search again.")
            continue

# Retrieve selected movie details
movie = df.loc[int(search_id)]

# Prompt user to choose basis for clustering and set the number of clusters
print("Choose the basis for clustering:\n1. Title\n2. Genre\n3. Both")
cluster_choice = int(input())
cluster_on = {1: 'title', 2: 'genre', 3: 'both'}[cluster_choice]

k = int(input("Enter the value of 'k' for clustering (must be greater than 2): "))
if k <= 2:
    print("Invalid 'k' value. Setting 'k' to 3 as default.")
    k = 3

# Perform clustering based on user's choice
if cluster_choice in [1, 2, 3]:
    df = kmeans_cluster(df, cluster_on=cluster_on, k=k)
    movie = df.loc[int(search_id)]
    cluster = df[df["cluster"] == movie["cluster"]].copy()
else:
    print("Invalid choice. Proceeding without clustering.")
    cluster = df.copy()

# User specifies the number of recommendations and weights for similarity measures
K = int(input("How many results would you like?: "))
desc_weight = float(input("Enter weight for description (0.0-1.0): "))
title_weight = float(input("Enter weight for title (0.0-1.0): "))
year_weight = float(input("Enter weight for year (0.0-1.0): "))

# Calculate similarity scores for each movie in the cluster based on user preferences
if 'description' in cluster.columns:
    cluster['cosine'] = cluster['description'].apply(lambda x: cosine_similarity_desc(movie['description'], x))
if 'title' in cluster.columns:
    cluster['levenshtein'] = cluster['title'].apply(lambda x: 1 - levenshtein_distance(movie['title'], x) / max(len(movie['title']), len(x)))
if 'year' in cluster.columns:
    cluster['euclidean'] = cluster['year'].apply(lambda x: euclidean_distance(movie['year'], x))
    cluster['euclidean'] = 1 - cluster['euclidean'] / cluster['euclidean'].max()

# Combine similarity scores using specified weights and select top recommendations
cluster['weight'] = desc_weight * cluster.get('cosine', 0) + title_weight * cluster.get('levenshtein', 0) + year_weight * cluster.get('euclidean', 0)

recommendations = cluster.sort_values(by='weight', ascending=False).head(K)
print(recommendations[['title', 'weight']])
