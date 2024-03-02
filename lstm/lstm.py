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

    def f_cur(self, x_cur: ArrayLike, h_prev: ArrayLike, c_prev: ArrayLike) -> ArrayLike:
        return sigmoid((self.wf @ jnp.concatenate([h_prev, x_cur], axis=1).T) + c_prev)
