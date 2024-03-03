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
        self.wf = rng_unif(key=self.key, shape=(hidden_dim, hidden_dim + input_dim))
        self.bf = rng_unif(key=self.key, shape=(hidden_dim,))
        self.wi = rng_unif(key=self.key, shape=(hidden_dim, hidden_dim + input_dim))
        self.bi = rng_unif(key=self.key, shape=(hidden_dim,))
        self.wc = rng_unif(key=self.key, shape=(hidden_dim, hidden_dim + input_dim))
        self.bc = rng_unif(key=self.key, shape=(hidden_dim,))
        self.wo = rng_unif(key=self.key, shape=(hidden_dim, hidden_dim + input_dim))
        self.bo = rng_unif(key=self.key, shape=(hidden_dim,))

    @jax.jit
    def f_cur(self, x_cur: ArrayLike, h_prev: ArrayLike) -> ArrayLike:
        return sigmoid((self.wf @ jnp.concatenate([h_prev, x_cur], axis=0)) + self.bf)

    @jax.jit
    def i_cur(self, x_cur: ArrayLike, h_prev: ArrayLike) -> tuple[ArrayLike, ArrayLike]:
        i_t = sigmoid((self.wi @ jnp.concatenate([h_prev, x_cur], axis=0)) + self.bi)
        c_t_hat = jnp.tanh((self.wc @ jnp.concatenate([h_prev, x_cur], axis=0)) + self.bc)
        return i_t, c_t_hat

    @jax.jit
    def c_cur(self, x_cur: ArrayLike, h_prev: ArrayLike, c_prev: ArrayLike) -> ArrayLike:
        i_t, c_t_hat = self.i_cur(x_cur, h_prev)
        return jnp.dot(self.f_cur(x_cur, h_prev), c_prev) + jnp.dot(i_t, c_t_hat)
    
    @jax.jit
    def o_cur(self, x_cur: ArrayLike, h_prev: ArrayLike) -> ArrayLike:
        return sigmoid((self.wo @ jnp.concatenate([h_prev, x_cur], axis=0)) + self.bo)
    
    @jax.jit
    def h_cur(self, x_cur: ArrayLike, h_prev: ArrayLike, c_prev: ArrayLike) -> ArrayLike:
        return jnp.dot(self.o_cur(x_cur, h_prev), jnp.tanh(self.c_cur(x_cur, h_prev, c_prev)))
    
    @jax.jit
    def forward(self, x_cur: ArrayLike, h_prev: ArrayLike, c_prev: ArrayLike) -> tuple[ArrayLike, ArrayLike]:
        return (self.c_cur(x_cur, h_prev, c_prev), self.h_cur(x_cur, h_prev, c_prev))

class PeepholeLSTM(LSTM):
    def __init__(self, seed: int, input_dim: int, hidden_dim: int):
        super().__init__(seed, input_dim, hidden_dim)
        self.wf = rng_unif(key=self.key, shape=(hidden_dim, 2 * hidden_dim + input_dim))
        self.wi = rng_unif(key=self.key, shape=(hidden_dim, 2 * hidden_dim + input_dim))
        self.wo = rng_unif(key=self.key, shape=(hidden_dim, 2 * hidden_dim + input_dim))
    
    @jax.jit
    def f_cur(self, x_cur: ArrayLike, h_prev: ArrayLike, c_prev: ArrayLike) -> ArrayLike:
        return sigmoid(self.wf @ jnp.concatenate([c_prev, h_prev, x_cur], axis=0) + self.bf)

    @jax.jit
    def i_cur(self, x_cur: ArrayLike, h_prev: ArrayLike, c_prev: ArrayLike) -> ArrayLike:
        return sigmoid(self.wi @ jnp.concatenate([c_prev, h_prev, x_cur], axis=0) + self.bi)
    
    @jax.jit
    def o_cur(self, x_cur: ArrayLike, h_prev: ArrayLike, c_cur: ArrayLike) -> ArrayLike:
        return sigmoid(self.wo @ jnp.concatenate([c_cur, h_prev, x_cur], axis=0) + self.bo)

    @jax.jit
    def h_cur(self, x_cur: ArrayLike, h_prev: ArrayLike, c_prev: ArrayLike, c_t: ArrayLike) -> ArrayLike:
        return jnp.dot(self.o_cur(x_cur, h_prev, c_t), jnp.tanh(self.c_cur(x_cur, h_prev, c_prev))) 

    @jax.jit
    def forward(self, x_cur: ArrayLike, h_prev: ArrayLike, c_prev: ArrayLike) -> tuple[ArrayLike, ArrayLike]:
        c_t = self.c_cur(x_cur, h_prev, c_prev)
        return (c_t, self.h_cur(x_cur, h_prev, c_prev, c_t))
