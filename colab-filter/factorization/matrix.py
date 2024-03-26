import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp

train_data = pd.read_csv("../preprocessed_dataset/train_dataset.csv")

print(train_data.head)
print(max(train_data["item_id_idx"]))

user_num = max(train_data["user_id_idx"]) + 1
item_num = max(train_data["item_id_idx"]) + 1

# A = feedback matrix
A = np.zeros((user_num, item_num))

for _, row in train_data.iterrows():
    A[row["user_id_idx"], row["item_id_idx"]] = row["rating"]

# d = embedding dimension
d = 100
w0 = 0.1

# U = user embedding matrix of size (user_num, d)
# V = item embedding matrix of size (item_num, d)
U = jnp.array(np.random.normal(0, 1, size=(user_num, d)))
V = jnp.array(np.random.normal(0, 1, size=(item_num, d)))


@jax.jit
def loss(A: jnp.ndarray, U: jnp.ndarray, V: jnp.ndarray):
    diff = A - U @ V.T
    return jnp.linalg.norm(diff)


@jax.jit
def weighted_loss(A: jnp.ndarray, U: jnp.ndarray, V: jnp.ndarray, w0):
    w = U @ V.T
    diff = A - w
    return jnp.linalg.norm(diff) + w0 * jnp.linalg.norm(w)


lr = 0.1


def gd_loss(A: jnp.ndarray, U: jnp.ndarray, V: jnp.ndarray):
    for i in range(100):
        U -= lr * jax.jit(jax.grad(loss, argnums=1)(A, U, V))
        V -= lr * jax.jit(jax.grad(loss, argnums=2)(A, U, V))
        if i % 10 == 0:
            print(loss(A, U, V))


print(loss(A, U, V))
print(weighted_loss(A, U, V, w0))
print(gd_loss(A, U, V))
