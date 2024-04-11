"# Movie-Recommendation-2-" 

About the Project 

Uses k-means to cluster the movies based on choices. You will then filter and sort based on additional metrics.

Instructions
For every recommendation query the following will happen:
1. Allows the user to choose any movie in the database. Provide an easy-to-use mechanism for this.
2. Upon choosing the movie it will create clusters of similar movies. What the clusters are based on depends on what factors the user thinks are important. Specifically, the user can choose either or both of the following attributes for the k-means algorithm (the user must choose at least one):
Title (some or all of it - you have creative license) (if you even want to use it - optional)
Genres
3. What the value of 'k' is depends on the user, but it must be greater than 2. Provide an easy-to-use mechanism for this.
Use either a GUI with check boxes, radio buttons, etc. or some text-based system that makes sense.

Based on the cluster that the movie is in then filter and sort the movies on that cluster. (The other clusters do not matter now for this query.) The user can choose any or all of the following to filter and sort the movies in the cluster:
- Uses cosine similarity on the description of the movies
- Uses Levenshtein distance for the titles
- Uses Euclidean distance with year as a factor for closest year

Also, the user can use weights to determine how much a particular attribute (e.g. cosine similarity of description) should count towards the final recommendation. For example, the user may decide that the cosine similarity should account for 80% of the filtering/sort, while the other two account for 10% each.
