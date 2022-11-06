import pandas as pd
from itertools import product
import numpy as np
import matplotlib.pyplot as plt
import sys
import streamlit as st

import warnings
warnings.filterwarnings("ignore")

st.title("Film Öneri Sistemi")

movieId = int(st.text_input('Insert MovieId'))

recomm_system = st.container()

with recomm_system:

    
    ratings = pd.read_csv('ml-latest-small/ratings.csv')
    # ratings.head(2)


    # In[50]:


    # ratings.shape


    # In[51]:


    # check if a user rate for a movie more than one time 
    temp = pd.DataFrame(ratings.groupby(['userId','movieId'])['timestamp'].value_counts())
    # temp.head(2)


    # In[52]:


    # check if a user rate for a movie more than one time : NO
    # temp['timestamp'].unique()


    # In[53]:


    links = pd.read_csv('ml-latest-small/links.csv')
    # links.head(2)


    # In[93]:


    movies = pd.read_csv('ml-latest-small/movies.csv')
    # movies.head(2)

    # if movieId not in movies['movieId'].unique():
    # print(f'Entered movieId ({movieId}) is not included in the movie list. Program will be ended.')
        # st.text(f'Entered movieId ({movieId}) is not included in the movie list. Program will be ended.')
        # sys.exit()

    # In[94]:


    movie_names_ids_dict = dict(zip(movies.title, movies.movieId))
    movie_ids_names_dict = dict(zip(movies.movieId, movies.title))


    # In[55]:


    tags = pd.read_csv('ml-latest-small/tags.csv')
    # tags.head(2)


    # In[56]:


    # tags['movieId'].nunique()


    # In[57]:


    movies['genres_list'] = movies.genres.str.split('|')

    # compute jaccard similarity between movies according to genres
    def jaccard_similarity(genres1, genres2):

        s1 = set(genres1)
        s2 = set(genres2)
        
        return float(len(s1.intersection(s2))/len(s1.union(s2)))


    # In[81]:

    # create a combination matrix for all movie pairs
    def create_product_matrix(col1, col2):
        
        prod = product(col1, col2)
        combinations = pd.DataFrame(list(prod),columns = ['movieId1','movieId2'])
        
        return combinations

    # compute the cosine similarity between movies according to user ratings
    def cosine_similarity(movie1, movie2):
    
        numerator = np.dot(np.where(np.isnan(movie1),0,movie1), np.where(np.isnan(movie2),0,movie2))
        m1_squared = np.dot(np.where(np.isnan(movie1),0,movie1), np.where(np.isnan(movie1),0,movie1))
        m2_squared = np.dot(np.where(np.isnan(movie2),0,movie2), np.where(np.isnan(movie2),0,movie2))
        denominator = np.sqrt(m1_squared * m2_squared)
        
        return numerator / denominator

    def film_oner(movies, ratings):

        movies_list = movies['movieId'].unique()
        movies = movies.set_index('movieId')
        jaccard_sim = {}

        for mov in movies_list:
            jaccard_sim[mov] = jaccard_similarity(movies.loc[movieId, :]['genres_list'],movies.loc[mov, :]['genres_list'])


        # In[62]:


        # jaccard_sim


        # In[65]:


        # Check if Jaccard scores extend between the interval of 0 and 1
        # jaccard_sim[min(jaccard_sim, key=jaccard_sim.get)], jaccard_sim[max(jaccard_sim, key=jaccard_sim.get)]


        # In[72]:


        jaccard_sim = pd.DataFrame(data = jaccard_sim.items(), columns = ['movieId2','jaccard_sim_score'])


        # In[68]:


        movies = movies.reset_index()


        # In[71]:


        related_movie = movies[movies['movieId'] == movieId]


        # In[86]:


        # Create a combination matrix having dimensions of [1 X (# of movies)]
        movie_pairs_mat = create_product_matrix(related_movie['movieId'], jaccard_sim['movieId2'])
        # movie_pairs_mat.head()


        # In[87]:


        movie_pairs_mat = movie_pairs_mat.merge(related_movie, left_on = 'movieId1', right_on = 'movieId', how = 'left')
        movie_pairs_mat = movie_pairs_mat.merge(jaccard_sim, on = 'movieId2', how = 'left')
        movie_pairs_mat.drop(columns = {'genres','movieId'}, inplace = True)

        ratings_pivot = ratings.pivot_table(index = 'movieId', columns = 'userId', values = 'rating')
        # ratings_pivot


        # In[99]:


        movies_list = ratings['movieId'].unique()
        rating_cos_sim = {}

        for mov in movies_list:
            rating_cos_sim[mov] = cosine_similarity(ratings_pivot.loc[movieId, :],ratings_pivot.loc[mov, :])


        # In[100]:


        # rating_cos_sim


        # In[102]:


        # Check lower and upper bounds, they extends between 0 and 1, no negative value, so everything is fine
        # rating_cos_sim[min(rating_cos_sim, key=rating_cos_sim.get)], rating_cos_sim[max(rating_cos_sim, key=rating_cos_sim.get)]


        # In[106]:


        cosine_sim = pd.DataFrame(data = rating_cos_sim.items(), columns = ['movieId2','cosine_sim_score'])
        # cosine_sim


        # In[107]:


        movie_pairs_mat = movie_pairs_mat.merge(cosine_sim, on = 'movieId2', how = 'left')


        # In[109]:


        # movie_pairs_mat.shape, movies.shape, ratings.shape


        # In[111]:


        movie_pairs_mat = movie_pairs_mat[movie_pairs_mat['movieId1'] != movie_pairs_mat['movieId2']]


        # In[112]:


        # movie_pairs_mat


        # In[126]:


        plt.scatter(movie_pairs_mat['jaccard_sim_score'],movie_pairs_mat['cosine_sim_score'])
        plt.xlabel('jaccard sim')
        plt.ylabel('cosine sim')


        # In[128]:


        ratings['rating'].hist()


        # In[116]:


        movie_pairs_mat['cumulative_score'] = movie_pairs_mat['jaccard_sim_score'] + movie_pairs_mat['cosine_sim_score']


        # In[119]:


        movie_pairs_mat['cumulative_score'].fillna(0, inplace = True)


        # In[120]:


        movie_pairs_mat = movie_pairs_mat.sort_values('cumulative_score', ascending = False)


        # In[122]:


        # movie_pairs_mat


        # In[130]:


        to_suggest = list(movie_pairs_mat.head(5)['movieId2'])
        # to_suggest

        return to_suggest

    # movie_ids_names_dict[suggest_movie_id]


    recommend_button = st.button('Get Recommendations')
    suggest_movie_ids = film_oner(movies, ratings)
    if recommend_button:
        st.header('You Entered the Movie:')
        st.text(f"{int(movieId)}: {movie_ids_names_dict[int(movieId)]}")
        st.header('Your Recommended Movies:')
        for mov in suggest_movie_ids:
            st.text(f"{mov}: {movie_ids_names_dict[mov]}")
