from time import time

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

train_data = pd.read_csv("../preprocessed_dataset/train_dataset.csv")

user_num = max(train_data["user_id_idx"]) + 1
item_num = max(train_data["item_id_idx"]) + 1

# A = feedback matrix
A = np.zeros((user_num, item_num))

for _, row in train_data.iterrows():
    A[row["user_id_idx"], row["item_id_idx"]] = row["rating"]


class MatrixFactorizationRecommender:
    def __init__(
        self,
        d=100,
        w0=0.1,
        lr=0.1,
        model="gd",
        steps=100,
        loss_threshold=10,
        batch_size=100,
    ) -> None:
        """
        d = embedding dimension
        w0 = regularization weight
        lr = learning rate
        model = factorization method
        steps = number of iterations
        loss_threshold = loss function threshold to stop iteration
        batch_size = batch size for stochastic gradient descent
        """
        self.d = d
        self.w0 = w0
        self.lr = lr
        self.model = model
        self.steps = steps
        self.threshold = loss_threshold
        self.batch_size = batch_size

    @jax.jit
    def loss(self, A: jnp.ndarray, U: jnp.ndarray, V: jnp.ndarray):
        w = U @ V.T
        diff = A - w
        return jnp.linalg.norm(diff) + self.w0 * jnp.linalg.norm(w)

    def gradient(self, fun, argnums):
        return jax.jit(jax.grad(fun, argnums=argnums))

    def gd(self, A: jnp.ndarray):
        for i in range(self.steps):
            self.U -= self.lr * self.gradient(self.loss, 1)(A, self.U, self.V)
            self.V -= self.lr * self.gradient(self.loss, 2)(A, self.U, self.V)
            if i % 10 == 0:
                print(f"Iteration {i}: loss {self.loss(A, self.U, self.V)}")
            if self.loss(A, self.U, self.V) < self.threshold:
                break

    def sgd(self, A: jnp.ndarray):
        def loss_vmap(argnums):
            return jax.jit(jax.vmap(jax.grad(self.loss, argnums)))

        for i in range(self.steps):
            self.U -= self.lr * loss_vmap(1)(A, self.U, self.V)
            self.V -= self.lr * loss_vmap(2)(A, self.U, self.V)
            if i % 10 == 0:
                print(f"Iteration {i}: loss {self.loss(A, self.U, self.V)}")
            if self.loss(A, self.U, self.V) < self.threshold:
                break

    def fit(self, A: np.ndarray):
        A = jnp.array(A)

        # U = user embedding matrix of size (user_num, d)
        # V = item embedding matrix of size (item_num, d)
        self.U = jnp.array(np.random.normal(0, 1, size=(user_num, self.d)))
        self.V = jnp.array(np.random.normal(0, 1, size=(item_num, self.d)))

        start_time = time()
        self.sgd(A)
        end_time = time()
        print(f"Elapsed time = {end_time - start_time} seconds")


r = MatrixFactorizationRecommender()
r.fit(A)
