import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from jax import Array

@jax.jit
def sigmoid(x: ArrayLike) -> ArrayLike:
    return 1./(1 + jnp.exp(-x))

@jax.jit
def rng_unif(key: Array, shape: tuple[int, int]) -> ArrayLike:
    return jax.random.uniform(key=key, shape=shape)

class LSTM:
    def __init__(self, seed: int, input_dim: int, hidden_dim: int):
        self.key = jax.random.PRNGKey(seed)
        self.wf = rng_unif(key=self.key, shape=(hidden_dim, input_dim))
        self.bf = rng_unif(key=self.key, shape=(hidden_dim,))
        self.wi = rng_unif(key=self.key, shape=(hidden_dim, input_dim))
        self.bi = rng_unif(key=self.key, shape=(hidden_dim,))
        self.wc = rng_unif(key=self.key, shape=(hidden_dim, input_dim))
        self.bc = rng_unif(key=self.key, shape=(hidden_dim,))

    def f_cur(self, x_cur: ArrayLike, h_prev: ArrayLike) -> ArrayLike:
        return sigmoid((self.wf @ jnp.concatenate([h_prev, x_cur], axis=1).T) + self.bf)

    def i_cur(self, x_cur: ArrayLike, h_prev: ArrayLike) -> tuple[ArrayLike, ArrayLike]:
        i_t = sigmoid((self.wi @ jnp.concatenate([h_prev, x_cur], axis=1).T) + self.bi)
        c_t_hat = jnp.tanh((self.wc @ jnp.concatenate([h_prev, x_cur], axis=1).T) + self.bc)
        return i_t, c_t_hat

    def c_cur(self, x_cur: ArrayLike, h_prev: ArrayLike, c_prev: ArrayLike) -> ArrayLike:
        i_t, c_t_hat = self.i_cur(x_cur, h_prev)
        return jnp.dot(self.f_cur(x_cur, h_prev), c_prev) + jnp.dot(i_t, c_t_hat)
    
    def o_cur(self, x_cur: ArrayLike, h_prev: ArrayLike) -> ArrayLike:
        pass
