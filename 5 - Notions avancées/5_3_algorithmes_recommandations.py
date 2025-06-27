import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity


# Fonction pour recommander des films
def recommend_movies(ratings_matrix, user_id, num_recommendations=5):
    # Trouver les utilisateurs similaires
    similar_users = user_similarity_df[user_id].sort_values(ascending=False).index[1:]  # Exclure l'utilisateur lui-même

    # Moyenne pondérée des notes des utilisateurs similaires
    similar_users_ratings = ratings_matrix.loc[similar_users]

    # Calculer la moyenne pondérée des évaluations pour chaque film
    weighted_ratings = similar_users_ratings.T.dot(user_similarity_df[user_id].sort_values(ascending=False)[1:])

    # Moyenne pondérée divisée par la somme des similarités pour normaliser
    weighted_ratings /= np.array([user_similarity_df[user_id].sort_values(ascending=False)[1:]]).sum()

    # Sélectionner les films que l'utilisateur n'a pas encore notés
    user_ratings = ratings_matrix.loc[user_id]
    unrated_movies = user_ratings[user_ratings == 0].index

    # Filtrer les films non évalués par l'utilisateur
    recommendations = weighted_ratings.loc[unrated_movies].sort_values(ascending=False).head(num_recommendations)

    return recommendations



if __name__ == '__main__':
    column_names = ['user_id', 'item_id', 'rating', 'timestamp']
    data = pd.read_csv('C:\\Users\\User\\PycharmProjects\\MachineLearning\\data\\movielens.csv', sep='\t', names=column_names)

    rating_matrix = data.pivot_table(index='user_id', columns='item_id', values='rating').fillna(0)

    user_similarity = cosine_similarity(rating_matrix)

    user_similarity_df = pd.DataFrame(user_similarity, index=rating_matrix.index, columns=rating_matrix.index)

    recommended_movies = recommend_movies(rating_matrix, user_id=5, num_recommendations=5)
    print("Films recommandés pour l'utilisateur 1:")
    print(recommended_movies)

