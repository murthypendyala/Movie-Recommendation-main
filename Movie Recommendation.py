# %%
import numpy as np
import pandas as pd

# %%
movies = pd.read_csv(r'E:\\code\\Bharat Intern\\Movie Recommendation\\movies.csv')
genome_scores = pd.read_csv(r'E:\\code\Bharat Intern\\Movie Recommendation\\links.csv')
tags = pd.read_csv(r'E:\\code\Bharat Intern\\Movie Recommendation\\tags.csv')
genome_tags = pd.read_csv(r'E:\\code\Bharat Intern\\Movie Recommendation\\links.csv')
ratings = pd.read_csv(r'E:\\code\Bharat Intern\\Movie Recommendation\\ratings.csv')

# %%
movies.head()

# %%
movies.shape

# %%
len(movies['movieId'].unique())

# %%
movies['genres'] = movies['genres'].str.replace('|', ' ')

# %%
movies.head()

# %%
genome_scores.head()

# %%
genome_tags.head()

# %%
genome_scores.shape

# %%
tags.head

# %%
tags.shape

# %%
genome_tags.shape

# %%
ratings.head()

# %%
ratings.shape

# %%
len(ratings['movieId'].unique())

# %%
#filtering and cleaning

ratings_f = ratings.groupby('userId').filter(lambda x: len(x) >= 55)

# %%
len(ratings_f['movieId'].unique()) / len(movies['movieId'].unique()) * 100

# %%
len(ratings_f['userId'].unique()) / len(ratings['userId'].unique()) * 100

# %%
movie_list_rating = ratings_f.movieId.unique().tolist()

# %%
movies = movies[movies['movieId'].isin(movie_list_rating)]

# %%
movies.head()

# %%
movies.shape

# %%
Mapping_file = dict(zip(movies['title'].tolist(), movies['movieId'].tolist()))

# %%
tags.drop(['timestamp'], axis = 1, inplace = True)
ratings_f.drop(['timestamp'], axis = 1, inplace = True)

# %%
#Merge the movies and tags Data Frames

mixed = pd.merge(movies, tags, on = 'movieId', how = 'left')
mixed.head()

# %%
#Create Metadata from genres and tag Columns

mixed.fillna("", inplace = True)
mixed = pd.DataFrame(mixed.groupby('movieId')['tag'].apply(lambda x: "%s" % ' '.join(x)))
Final = pd.merge(movies, mixed, on = 'movieId', how = 'left')
Final['metadata'] = Final[['tag', 'genres']].apply(lambda x: ' '.join(x), axis = 1)
Final[['movieId','title','metadata']].head()

# %%
#Creating a content latent matrix from movie metadata:

from sklearn.feature_extraction.text import TfidfVectorizer

# %%
tfidf = TfidfVectorizer(stop_words = 'english')
tfidf_matrix = tfidf.fit_transform(Final['metadata'])
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), index = Final.index.tolist())
print(tfidf_df.shape)

# %%
from sklearn.decomposition import TruncatedSVD    

# %%
svd = TruncatedSVD(n_components = 200)
latent_matrix = svd.fit_transform(tfidf_df)

# %%
import matplotlib.pyplot as plt
explained = svd.explained_variance_ratio_.cumsum()
plt.plot(explained, '.-', ms = 16, color ='red')
plt.xlabel('Singular value components', fontsize = 12)
plt.ylabel('Cumulative percent of variance', fontsize = 12)        
plt.show()

# %%
n = 200
latent_matrix_1_df = pd.DataFrame(latent_matrix[:,0:n], index=Final.title.tolist())
latent_matrix_1_df.shape

# %%
#Creating a collaborative latent matrix from user ratings:

ratings_f.head()

# %%
ratings_f1 = pd.merge(movies[['movieId']], ratings_f, on = "movieId", how = "right")


# %%
ratings_f2 = ratings_f1.pivot(index = 'movieId', columns = 'userId', values = 'rating').fillna(0)
ratings_f2.head()


# %%
svd = TruncatedSVD(n_components = 200)
latent_matrix_2 = svd.fit_transform(ratings_f2)
latent_matrix_2_df = pd.DataFrame(latent_matrix_2, index = Final.title.tolist())


# %%
explained = svd.explained_variance_ratio_.cumsum()
plt.plot(explained, '.-', ms = 16, color = 'red')
plt.xlabel('Singular value components', fontsize = 12)
plt.ylabel('Cumulative percent of variance', fontsize = 12)        
plt.show()


# %%
#Content/Collaborative and Hybrid cosine similarity

from sklearn.metrics.pairwise import cosine_similarity, linear_kernel


# %%
a_1 = np.array(latent_matrix_1_df.loc["Look Who's Talking (1989)"]).reshape(1, -1)
a_2 = np.array(latent_matrix_2_df.loc["Look Who's Talking (1989)"]).reshape(1, -1)


# %%
score_1 = cosine_similarity(latent_matrix_1_df, a_1).reshape(-1)
score_2 = cosine_similarity(latent_matrix_2_df, a_2).reshape(-1)


# %%
hybrid = ((score_1 + score_2) / 2.0)


# %%
dictDf = {'content': score_1, 'collaborative': score_2, 'hybrid': hybrid} 
similar = pd.DataFrame(dictDf, index = latent_matrix_1_df.index)



# %%
similar.sort_values('content', ascending = False, inplace = True)
similar[1:].head(11)

# %%
b_1 = np.array(latent_matrix_1_df.loc["Look Who's Talking (1989)"]).reshape(1, -1)
b_2 = np.array(latent_matrix_2_df.loc["Look Who's Talking (1989)"]).reshape(1, -1)


# %%
score_1_lin = linear_kernel(latent_matrix_1_df, b_1).reshape(-1)
score_2_lin = linear_kernel(latent_matrix_2_df, b_2).reshape(-1)


# %%
hybrid = ((score_1_lin + score_2_lin) / 2.0)


# %%
dictDf = {'content': score_1_lin, 'collaborative': score_2_lin, 'hybrid': hybrid} 
similar = pd.DataFrame(dictDf, index = latent_matrix_1_df.index)


# %%
similar.sort_values('content', ascending = False, inplace = True)
similar[1:].head(11)


# %%



