import warnings

warnings.filterwarnings('ignore')
import numpy as np

import pandas as pd
from sklearn.neighbors import NearestNeighbors


class RecommendationEngine:

    k: int

    def __init__(self, data_file):
        self.data = pd.read_csv(data_file, sep=';', error_bad_lines=False, encoding="latin-1", dtype=object)
        self.metric = 'cosine'
        self.k = 3

    def find_similar_users(self, user_id, k=None):
        if k is None:
           k = self.k

        model_knn = NearestNeighbors(metric=self.metric, algorithm='brute')
        data_without_itemid = self.data.drop('ItemId', axis=1)
        model_knn.fit(data_without_itemid)

        loc = self.data.loc[self.data['UserId'] == user_id].index[0]
        distances, indices = model_knn.kneighbors(self.data.iloc[loc, 1:].values.reshape(1, -1), n_neighbors=k + 1)
        similarities = 1 - distances.flatten()

        return similarities, indices

    def find_similar_items(self, item_id, k=None):
        if k is None:
            k = self.k

        ratings = self.data.pivot(index='UserId', columns='ItemId', values='Rating')
        similarities = []
        indices = []

        if item_id in ratings.columns:
            model_knn = NearestNeighbors(metric=self.metric, algorithm='brute')
            model_knn.fit(ratings.T)
            loc = ratings.columns.get_loc(item_id)
            distances, indices = model_knn.kneighbors(ratings.iloc[:, loc].values.reshape(1, -1), n_neighbors=k + 1)
            similarities = 1 - distances.flatten()

        return similarities, indices

    def predict_userbased(self, user_id, item_id):
        ratings = self.data.pivot(index='UserId', columns='ItemId', values='Rating')
        mean_rating = ratings.loc[user_id].mean()
        similarities, indices = self.find_similar_users(user_id)
        sum_wt = similarities.sum() - 1
        product = 1
        wtd_sum = 0

        for i in range(1, len(indices.flatten())):
            if indices.flatten()[i] == user_id:
                continue
            else:
                ratings_diff = ratings.iloc[indices.flatten()[i], ratings.columns.get_loc(item_id)] - ratings.iloc[
                    indices.flatten()[i]].mean()
                product = ratings_diff * (similarities[i])
                wtd_sum = wtd_sum + product

        prediction = int(round(mean_rating + (wtd_sum / sum_wt)))

        if prediction <= 0:
            prediction = 1
        elif prediction > 10:
            prediction = 10

        return prediction

    def predict_itembased(self, user_id, item_id):
        ratings = self.data.pivot(index='UserId', columns='ItemId', values='Rating')
        similarities, indices = self.find_similar_items(item_id)
        sum_wt = similarities.sum() - 1
        product = 1
        wtd_sum = 0

        for i in range(1, len(indices.flatten())):
            if indices.flatten()[i] == ratings.columns.get_loc(item_id):
                continue
            else:
                product = ratings.iloc[ratings.index.get_loc(user_id), indices.flatten()[i]] * (similarities[i])
                wtd_sum = wtd_sum + product

        prediction = int(round(wtd_sum / sum_wt))

        if prediction <= 0:
            prediction = 1
        elif prediction > 10:
            prediction = 10

        return prediction