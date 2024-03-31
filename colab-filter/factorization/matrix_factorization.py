from time import time

import jax
import jax.numpy as jnp
import numpy as np


@jax.jit
def norm(x):
    return jnp.sum(jnp.square(x))


@jax.jit
def _loss(A, U, V, w0):
    A, U, V = jnp.array(A), jnp.array(U), jnp.array(V)
    diff = A - U @ V.T
    return norm(diff) + w0 * (norm(U) + norm(V))


class MatrixFactorizationRecommender:
    def __init__(
        self,
        d=10,
        w0=0.1,
        lr=0.001,
        model="gd",
        steps=100,
        batch_size=200,
        k=10,
    ) -> None:
        """
        d = embedding dimension
        w0 = regularization weight
        lr = learning rate
        model = factorization method ('gd', 'sgd', 'svd', 'als', 'als_solve')
        steps = number of iterations
        batch_size = batch size for stochastic gradient descent
        k = features for SVD
        """
        self.d = d
        self.w0 = w0
        self.lr = lr
        self.model = model
        self.steps = steps
        self.batch_size = batch_size
        self.k = k
        self.prediction = None

    def loss(self, A: np.ndarray, U: np.ndarray, V: np.ndarray):
        """
        Objective loss function we want to optimize.
        """
        n, m = A.shape
        w = U @ V.T
        loss = 0
        for i in range(n):
            for j in range(m):
                if A[i, j] != 0:
                    loss += np.sum(np.square(A[i, j] - w[i, j]))
        return loss + self.w0 * (np.sum(np.square(U)) + np.sum(np.square(V)))

    def gradient(self, fun, argnums):
        return jax.jit(jax.grad(fun, argnums=argnums))

    def gd(self, A: np.ndarray):
        """
        Implementation of Gradient Descent using JAX gradient.
        """

        def gd_loss(A, U, V):
            return _loss(A, U, V, self.w0)

        A = jnp.array(A)
        self.U, self.V = jnp.array(self.U), jnp.array(self.V)
        for step in range(self.steps):
            self.U -= self.lr * self.gradient(gd_loss, 1)(A, self.U, self.V)
            self.V -= self.lr * self.gradient(gd_loss, 2)(A, self.U, self.V)

    def sgd(self, A: np.ndarray):
        """
        Implementation of Mini Batch Stochastic Gradient Descent.
        """
        for step in range(self.steps):
            uu = np.random.choice(A.shape[0], (self.batch_size,))
            ii = np.random.choice(A.shape[1], (self.batch_size,))
            for j in range(self.batch_size):
                u, i = uu[j], ii[j]
                err = A[u, i] - self.U[u, :] @ self.V.T[:, i]
                self.U[u, :] += (
                    self.lr * 2 * (err * self.V[i, :] - self.w0 * self.U[u, :])
                ) / self.d
                self.V[i, :] += (
                    self.lr * 2 * (err * self.U[u, :] - self.w0 * self.V[i, :])
                ) / self.d

    def svd(self, A: np.ndarray):
        """
        Implementation of Singular Value Decomposition as initial embeddings then
        optimize using SGD.
        """
        u, sigma, vt = np.linalg.svd(A)
        if self.k != -1:
            u = u[:, : self.k]
            vt = vt[: self.k, :]
            self.sigma = np.eye(self.k) * sigma[: self.k]
        else:
            self.sigma = np.zeros((5000, 20361))
            for i in range(sigma.shape[0]):
                self.sigma[i, i] = sigma[i]
        self.U = u.astype(float) @ self.sigma
        self.V = vt.T.astype(float)

        for step in range(self.steps):
            uu = np.random.choice(A.shape[0], (self.batch_size,))
            ii = np.random.choice(A.shape[1], (self.batch_size,))
            for j in range(self.batch_size):
                u, i = uu[j], ii[j]
                err = A[u, i] - self.U[u, :] @ self.V.T[:, i]
                self.U[u, :] += (
                    self.lr * 2 * (err * self.V[i, :] - self.w0 * self.U[u, :])
                ) / self.d
                self.V[i, :] += (
                    self.lr * 2 * (err * self.U[u, :] - self.w0 * self.V[i, :])
                ) / self.d

    def als(self, A: np.ndarray):
        """
        Implementation of Alternating Least Squares using np.linalg.inv().
        """
        for step in range(self.steps):
            # fix V
            self.U = (
                A
                @ self.V
                @ (np.linalg.inv(self.V.T @ self.V + self.w0 * np.identity(self.d)))
            )
            # fix U
            self.V = (
                A.T
                @ self.U
                @ np.linalg.inv(self.U.T @ self.U + self.w0 * np.identity(self.d))
            )

    def als_solve(self, A: np.ndarray):
        """
        Implementation of Alternating Least Squares using np.solve().
        """
        for step in range(self.steps):
            # fix V
            self.U = np.linalg.solve(
                (self.V.T @ self.V + self.w0 * np.identity(self.d)), self.V.T @ A.T
            ).T
            # fix U
            self.V = np.linalg.solve(
                (self.U.T @ self.U + self.w0 * np.identity(self.d)), self.U.T @ A
            ).T

    def fit(self, train_data):
        """
        Fit the model by learning from given training data.
        """
        user_num = max(train_data["user_id_idx"]) + 1
        item_num = max(train_data["item_id_idx"]) + 1

        # A = feedback matrix
        A = np.zeros((user_num, item_num))

        for _, row in train_data.iterrows():
            A[row["user_id_idx"], row["item_id_idx"]] = row["rating"]

        # U = user embedding matrix of size (user_num, d)
        # V = item embedding matrix of size (item_num, d)
        self.U = np.random.normal(0, 1, size=(user_num, self.d))
        self.V = np.random.normal(0, 1, size=(item_num, self.d))

        print(f"Initial loss: {self.loss(A, self.U, self.V)}")
        start_time = time()
        if self.model == "gd":
            self.gd(A)
        elif self.model == "sgd":
            self.sgd(A)
        elif self.model == "svd":
            self.svd(A)
        elif self.model == "als":
            self.als(A)
        elif self.model == "als_solve":
            self.als_solve(A)
        else:
            raise ValueError("Invalid model specified")
        end_time = time()
        print(f"Elapsed time = {end_time - start_time} seconds")
        print(f"Final loss: {self.loss(A, self.U, self.V)}")
        del A

    def predict_top_k(self, user_idx, k=3):
        """
        Given a user index and k, return top k recommended songs for the user.
        """
        if self.prediction is None:
            self.prediction = self.U @ self.V.T

        ratings = self.prediction[user_idx, :]
        ratings = [(i, ratings[i]) for i in range(len(ratings))]
        ratings.sort(key=lambda x: x[1], reverse=True)
        return [i[0] for i in ratings[:k]]

    def evaluate_top_k(self, test_data):
        """
        Given test data, evaluate the model performance by giving top 3 song recommendations
        to users in the test data.
        A rating is considered positive if rating >= 3.
        True Positive if the recommendation is in the test data.
        False Positive if the recommendation is not in the test data.
        """
        TP, FP = 0, 0

        test = {}
        for _, row in test_data.iterrows():
            item_idx = row["item_id_idx"]
            user_idx = row["user_id_idx"]
            rating = row["rating"]
            if rating >= 3:
                if user_idx not in test:
                    test[user_idx] = []
                test[user_idx].append(item_idx)

        prediction = self.U @ self.V.T
        prediction = np.fliplr(np.argsort(prediction))
        for user_idx in sorted(list(test.keys())):
            ratings = prediction[user_idx, :][:3]
            for p in ratings:
                if p in test[user_idx]:
                    TP += 1
                else:
                    FP += 1

        precision = TP / (TP + FP) if TP + FP > 0 else 0
        recall = TP / test_data.shape[0] if test_data.shape[0] > 0 else 0
        return {
            "TP": TP,
            "FP": FP,
            "precision": precision,
            "recall": recall,
            "F1": (
                2 * precision * recall / (precision + recall)
                if precision + recall > 0
                else 0
            ),
        }

    def evaluate(self, test_data):
        """
        Given test data, evaluate the model performance by predicting
        the song ratings for users and songs in the test data.
        A rating is considered positive if rating >= 3.
        True Positive if both prediction and test data have positive ratings.
        False Positive if positive prediction but negative in test data.
        False Negative if negative prediction but positive in test data.
        True Negative if both prediction and test data have negative ratings.
        """
        prediction = self.U @ self.V.T

        TP, TN, FP, FN = 0, 0, 0, 0

        for _, row in test_data.iterrows():
            item_idx = row["item_id_idx"]
            user_idx = row["user_id_idx"]
            rating = row["rating"]
            pred = prediction[user_idx, item_idx]
            if pred >= 3 and rating >= 3:
                TP += 1
            elif pred >= 3 and rating < 3:
                FP += 1
            elif pred < 3 and rating >= 3:
                FN += 1
            else:
                TN += 1

        precision = TP / (TP + FP) if TP + FP > 0 else 0
        recall = TP / (TP + FN) if TP + FN > 0 else 0
        return {
            "TP": TP,
            "FP": FP,
            "FN": FN,
            "TN": TN,
            "precision": precision,
            "recall": recall,
            "F1": (
                2 * precision * recall / (precision + recall)
                if precision + recall > 0
                else 0
            ),
        }
