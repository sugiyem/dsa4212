import jax 
import jax.numpy as jnp
import numpy as np 
import pandas as pd 
import heapq 

@jax.jit
def dot_product(x, y):
    return jnp.sum(x * y)

@jax.jit 
def l2_norm(x):
    return jnp.sqrt(dot_product(x, x))

# Implementation of recommender in chapter 8 of "Practical Recommender System"
# The idea is to find n most similar user, then get the best rated song from these user
class BasicClusteringRecommender:
    # n_neighbor will specify the number of most similar user to find
    # n_recommendation will specify the number of recommended songs to output
    def __init__(self, n_neighbor, n_recommendation=10):
        self.n_neighbor = n_neighbor
        self.n_recommendation = n_recommendation
        self.similar_user_dict = {}

    # data must be a pd.dataframe with at least 3 columns, userID songID and rating
    def fit(self, data):
        self.n_userID = data.userID.max() + 1
        self.n_songID = data.songID.max() + 1
        self.data = data

    def get_songs_and_ratings(self, userID):
        filtered_data = self.data[self.data.userID == userID]
        return filtered_data.songID.values, filtered_data.rating.values
    
    # given userID, find n_neighbor most similar users
    def get_similar_users(self, userID):
        if userID in self.similar_user_dict:
            return self.similar_user_dict[userID]
        
        userSongs, userRatings = self.get_songs_and_ratings(userID)
        userNorm = l2_norm(userRatings)
        
        userVector = np.zeros(self.n_songID)
        userVector[userSongs] = userRatings

        pq = []

        for i in range(self.n_userID):
            if i == userID:
                continue 

            currSongs, currRatings = self.get_songs_and_ratings(i)

            similarity_score = dot_product(userVector[currSongs], currRatings) / (userNorm * l2_norm(currRatings))

            heapq.heappush(pq, (similarity_score, i))
            
            if len(pq) > self.n_neighbor:
                heapq.heappop(pq)

        similar_users = [similarUserID for _, similarUserID in pq]
        self.similar_user_dict[userID] = similar_users
        return similar_users

    # given userID and songID, predict the given rating
    def predict_score(self, userID, songID):
        similar_users = self.get_similar_users(userID)
        
        cnt, sum = 0, 0
        for similarUserID in similar_users:
            similarSongs, similarRatings = self.get_songs_and_ratings(similarUserID)
            indices = np.where(similarSongs == songID)[0]
            if len(indices) == 0:
                continue 

            cnt += 1
            sum += similarRatings[indices[0]]

        if sum == 0:
            # default rating if no similar user rates this song
            return 3
        else:
            return round(sum / cnt)
    
    # given userID, find the recommendation for this user
    def predict(self, userID):
        userSongs, _ = self.get_songs_and_ratings(userID)
        similar_users = self.get_similar_users(userID)
        
        totalSum = np.zeros(self.n_songID)
        totalCnt = np.full(self.n_songID, 1e-6)
        for similarUserID in similar_users:
            similarSongs, similarRatings = self.get_songs_and_ratings(similarUserID)
            totalSum[similarSongs] += similarRatings
            totalCnt[similarSongs] += 1.
        avg = totalSum / totalCnt

        # remove user's rated song
        avg[userSongs] = 0

        # get the best rated song
        return np.argsort(avg)[-self.n_recommendation:]
    
# Implementation of recommender in chapter 7 + 8 of "Practical Recommender System"
# The idea is to use minibatch K-means clustering to split the user to a number of clusters
# Then, each time we want to predict something, we'll find n most similar user in the same cluster
class KMeansRecommender():
    def __init__(self, n_neighbor, n_recommendation=10, n_cluster=10, use_minibatch=True):
        self.n_neighbor = n_neighbor
        self.n_recommendation = n_recommendation
        self.similar_user_dict = {}
        self.n_cluster = n_cluster
        self.use_minibatch = use_minibatch

    def get_songs_and_ratings(self, userID):
        filtered_data = self.data[self.data.userID == userID]
        return filtered_data.songID.values, filtered_data.rating.values
    
    # data must be a pd.dataframe with 3 columns, userID songID rating
    def fit(self, data):
        self.n_userID = data.userID.max() + 1
        self.n_songID = data.songID.max() + 1
        self.data = data

        # perform mini-batch k-means clustering
        n_iter = 10
        n_samples = self.n_userID // 5 # to be used in minibatch 

        # iteratively update the centroid
        centroids = np.random.randint(3, 6, (self.n_cluster, self.n_songID))
        for _ in range(n_iter):
            indices = np.arange(self.n_userID)
            if self.use_minibatch:
                indices = np.random.choice(self.n_userID, n_samples, replace=True)
            
            centroid_norms = [l2_norm(centroid - 3) for centroid in centroids]

            new_centroids = np.zeros_like(centroids)
            nearest_centroid_cnt = np.full(self.n_cluster, 1e-6)
            for idx in indices:
                currSongs, currRatings = self.get_songs_and_ratings(idx)
                dists = [centroid_norms[i] - l2_norm(centroids[i, currSongs] - 3) + l2_norm(currRatings - centroids[i, currSongs]) for i in range(self.n_cluster)]
                nearest_centroid_idx = np.argmin(dists)

                nearest_centroid_cnt[nearest_centroid_idx] += 1.0
                new_centroids[nearest_centroid_idx, currSongs] += currRatings

            centroids = new_centroids / nearest_centroid_cnt[:, np.newaxis]

        # problem: in current implementation, most centroid will converge to [0, 0, ...., 0] due to sparsity of data
        
        # choose the nearest centroid for each user
        self.nearest_centroid = [-1 for _ in range(self.n_userID)]
        self.cluster = [[] for _ in range(self.n_cluster)]
        centroid_norms = [l2_norm(centroid - 3) for centroid in centroids]
        for userID in range(self.n_userID):
            userSongs, userRatings = self.get_songs_and_ratings(userID)
            dists = [centroid_norms[i] - l2_norm(centroids[i, userSongs] - 3) + l2_norm(userRatings - centroids[i, userSongs]) for i in range(self.n_cluster)]
            centroid_idx = np.argmin(dists)
            self.nearest_centroid[userID] = centroid_idx
            self.cluster[centroid_idx].append(userID)
    
    # given userID, find n_neighbor most similar users
    def get_similar_users(self, userID):
        if userID in self.similar_user_dict:
            return self.similar_user_dict[userID]
        
        userSongs, userRatings = self.get_songs_and_ratings(userID)
        userNorm = l2_norm(userRatings)
        
        userVector = np.zeros(self.n_songID)
        userVector[userSongs] = userRatings
        
        clustered_users = self.cluster[self.nearest_centroid[userID]]
        pq = []

        for i in clustered_users:
            if i == userID:
                continue 

            currSongs, currRatings = self.get_songs_and_ratings(i)

            similarity_score = dot_product(userVector[currSongs], currRatings) / (userNorm * l2_norm(currRatings))

            heapq.heappush(pq, (similarity_score, i))
            
            if len(pq) > self.n_neighbor:
                heapq.heappop(pq)

        similar_users = [similarUserID for _, similarUserID in pq]
        self.similar_user_dict[userID] = similar_users
        return similar_users
    
    # given userID and songID, predict the given rating
    def predict_score(self, userID, songID):
        similar_users = self.get_similar_users(userID)
        
        cnt, sum = 0, 0
        for similarUserID in similar_users:
            similarSongs, similarRatings = self.get_songs_and_ratings(similarUserID)
            indices = np.where(similarSongs == songID)[0]
            if len(indices) == 0:
                continue 

            cnt += 1
            sum += similarRatings[indices[0]]

        if sum == 0:
            # default rating if no similar user rates this song
            return 3
        else:
            return round(sum / cnt)
    
    # given userID, find the recommendation for this user
    def predict(self, userID):
        userSongs, _ = self.get_songs_and_ratings(userID)
        similar_users = self.get_similar_users(userID)
        
        totalSum = np.zeros(self.n_songID)
        totalCnt = np.full(self.n_songID, 1e-6)
        for similarUserID in similar_users:
            similarSongs, similarRatings = self.get_songs_and_ratings(similarUserID)
            totalSum[similarSongs] += similarRatings
            totalCnt[similarSongs] += 1.
        avg = totalSum / totalCnt

        # remove user's rated song
        avg[userSongs] = 0

        # get the best rated song
        return np.argsort(avg)[-self.n_recommendation:]