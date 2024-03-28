from time import time

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

train_data = pd.read_csv("./dataset/train_dataset.csv")

user_num = max(train_data["user_id_idx"]) + 1
item_num = max(train_data["item_id_idx"]) + 1

# A = feedback matrix
A = np.zeros((user_num, item_num))

for _, row in train_data.iterrows():
    A[row["user_id_idx"], row["item_id_idx"]] = row["rating"]


@jax.jit
def norm(x):
    return jnp.linalg.norm(x)


@jax.jit
def _loss(A, U, V, w0):
    A, U, V = jnp.array(A), jnp.array(U), jnp.array(V)
    diff = A - U @ V.T
    return norm(diff) + w0 * (norm(U) + norm(V))


class MatrixFactorizationRecommender:
    def __init__(
        self, d=100, w0=0.1, lr=0.1, model="gd", steps=100, loss_threshold=10, k=-1
    ) -> None:
        """
        d = embedding dimension
        w0 = regularization weight
        lr = learning rate
        model = factorization method ('gd', 'svd')
        steps = number of iterations
        loss_threshold = loss function threshold to stop iteration
        k = features for SVD
        """
        self.d = d
        self.w0 = w0
        self.lr = lr
        self.model = model
        self.steps = steps
        self.threshold = loss_threshold
        self.k = k

    def loss(self, A: jnp.ndarray, U: jnp.ndarray, V: jnp.ndarray):
        return _loss(A, U, V, self.w0)

    def gradient(self, fun, argnums):
        return jax.jit(jax.grad(fun, argnums=argnums))

    def gd(self, A: np.ndarray):
        A = jnp.array(A)
        for i in range(self.steps):
            self.U -= self.lr * self.gradient(self.loss, 1)(A, self.U, self.V)
            self.V -= self.lr * self.gradient(self.loss, 2)(A, self.U, self.V)
            if i % 10 == 0:
                print(f"Iteration {i}: loss {self.loss(A, self.U, self.V)}")
            if self.loss(A, self.U, self.V) < self.threshold:
                break

    def svd(self, A: np.ndarray):
        u, sigma, vt = np.linalg.svd(A)
        if self.k != -1:
            u = u[:, : self.k]
            vt = vt[: self.k, :]
            sigma = np.eye(self.k) * sigma[: self.k]
        self.U = u.astype(float).reshape(-1, 1)
        self.sigma = sigma.astype(float).reshape(-1, 1)
        self.V = vt.T.astype(float).reshape(-1, 1)
        print(self.U.dtype, self.V.dtype, self.sigma.dtype)
        print(self.loss(A, self.U, self.V))

    def fit(self, A: np.ndarray):
        # U = user embedding matrix of size (user_num, d)
        # V = item embedding matrix of size (item_num, d)
        self.U = jnp.array(np.random.normal(0, 1, size=(user_num, self.d)))
        self.V = jnp.array(np.random.normal(0, 1, size=(item_num, self.d)))

        start_time = time()
        if self.model == "gd":
            self.gd(A)
        elif self.model == "svd":
            self.svd(A)
        else:
            raise ValueError("Invalid model specified")
        end_time = time()
        print(f"Elapsed time = {end_time - start_time} seconds")


r = MatrixFactorizationRecommender(model="gd")
r.fit(A)
