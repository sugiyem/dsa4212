import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from jax import Array

RANDOM_SEED = 42

@jax.jit
def sigmoid(x: ArrayLike) -> ArrayLike:
    return 1./(1 + jnp.exp(-x))

def rng_unif(key: Array, shape: tuple[int, int]) -> ArrayLike:
    return jax.random.uniform(key=key, shape=shape)

class LSTM:
    def __init__(
        self, 
        seed: int, 
        input_dim: int, 
        hidden_dim: int
    ):
        self.key = jax.random.PRNGKey(seed)
        self.wf = rng_unif(key=self.key, shape=(hidden_dim, hidden_dim + input_dim))
        self.bf = rng_unif(key=self.key, shape=(hidden_dim,))
        self.wi = rng_unif(key=self.key, shape=(hidden_dim, hidden_dim + input_dim))
        self.bi = rng_unif(key=self.key, shape=(hidden_dim,))
        self.wc = rng_unif(key=self.key, shape=(hidden_dim, hidden_dim + input_dim))
        self.bc = rng_unif(key=self.key, shape=(hidden_dim,))
        self.wo = rng_unif(key=self.key, shape=(hidden_dim, hidden_dim + input_dim))
        self.bo = rng_unif(key=self.key, shape=(hidden_dim,))

    def f_cur(self, x_cur: ArrayLike, h_prev: ArrayLike) -> ArrayLike:
        @jax.jit
        def f_cur_jit(x_cur: ArrayLike, h_prev: ArrayLike) -> ArrayLike:
            return sigmoid((self.wf @ jnp.concatenate([h_prev, x_cur], axis=0)) + self.bf)
        return f_cur_jit(x_cur, h_prev)

    
    def i_cur(self, x_cur: ArrayLike, h_prev: ArrayLike) -> tuple[ArrayLike, ArrayLike]:
        @jax.jit
        def i_cur_jit(x_cur: ArrayLike, h_prev: ArrayLike) -> tuple[ArrayLike, ArrayLike]:
            i_t = sigmoid((self.wi @ jnp.concatenate([h_prev, x_cur], axis=0)) + self.bi)
            c_t_hat = jnp.tanh((self.wc @ jnp.concatenate([h_prev, x_cur], axis=0)) + self.bc)
            return i_t, c_t_hat
        return i_cur_jit(x_cur, h_prev)

    def c_cur(self, x_cur: ArrayLike, h_prev: ArrayLike, c_prev: ArrayLike) -> ArrayLike:
        @jax.jit
        def c_cur_jit(x_cur: ArrayLike, h_prev: ArrayLike, c_prev: ArrayLike) -> ArrayLike:
            i_t, c_t_hat = self.i_cur(x_cur, h_prev)
            return jnp.dot(self.f_cur(x_cur, h_prev), c_prev) + jnp.dot(i_t, c_t_hat)
        return c_cur_jit(x_cur, h_prev, c_prev)
        
    
    def o_cur(self, x_cur: ArrayLike, h_prev: ArrayLike) -> ArrayLike:
        @jax.jit
        def o_cur_jit(x_cur: ArrayLike, h_prev: ArrayLike) -> ArrayLike:
            return sigmoid((self.wo @ jnp.concatenate([h_prev, x_cur], axis=0)) + self.bo)
        return o_cur_jit(x_cur, h_prev)
    
    def h_cur(self, x_cur: ArrayLike, h_prev: ArrayLike, c_prev: ArrayLike) -> ArrayLike:
        @jax.jit
        def h_cur_jit(x_cur: ArrayLike, h_prev: ArrayLike, c_prev: ArrayLike) -> ArrayLike:
            return jnp.dot(self.o_cur(x_cur, h_prev), jnp.tanh(self.c_cur(x_cur, h_prev, c_prev)))
        return h_cur_jit(x_cur, h_prev, c_prev)
        
    def forward(self, x_cur: ArrayLike, h_prev: ArrayLike, c_prev: ArrayLike) -> tuple[ArrayLike, ArrayLike]:
        @jax.jit
        def forward_jit(x_cur: ArrayLike, h_prev: ArrayLike, c_prev: ArrayLike) -> tuple[ArrayLike, ArrayLike]:
            return (self.c_cur(x_cur, h_prev, c_prev), self.h_cur(x_cur, h_prev, c_prev))
        return forward_jit(x_cur, h_prev, c_prev)

class PeepholeLSTM(LSTM):
    def __init__(
        self, 
        seed: int, 
        input_dim: int, 
        hidden_dim: int
    ):
        super().__init__(seed, input_dim, hidden_dim)
        self.wf = rng_unif(key=self.key, shape=(hidden_dim, 2 * hidden_dim + input_dim))
        self.wi = rng_unif(key=self.key, shape=(hidden_dim, 2 * hidden_dim + input_dim))
        self.wo = rng_unif(key=self.key, shape=(hidden_dim, 2 * hidden_dim + input_dim))
    
    
    def f_cur(self, x_cur: ArrayLike, h_prev: ArrayLike, c_prev: ArrayLike) -> ArrayLike:
        @jax.jit
        def f_cur_jit(x_cur: ArrayLike, h_prev: ArrayLike, c_prev: ArrayLike) -> ArrayLike:
            return sigmoid(self.wf @ jnp.concatenate([c_prev, h_prev, x_cur], axis=0) + self.bf)
        return f_cur_jit(x_cur, h_prev, c_prev)

    
    def i_cur(self, x_cur: ArrayLike, h_prev: ArrayLike, c_prev: ArrayLike) -> ArrayLike:
        @jax.jit
        def i_cur_jit(x_cur: ArrayLike, h_prev: ArrayLike, c_prev: ArrayLike) -> ArrayLike:
            return sigmoid(self.wi @ jnp.concatenate([c_prev, h_prev, x_cur], axis=0) + self.bi)
        return i_cur_jit(x_cur, h_prev, c_prev)
    
    def o_cur(self, x_cur: ArrayLike, h_prev: ArrayLike, c_cur: ArrayLike) -> ArrayLike:
        @jax.jit
        def o_cur_jit(x_cur: ArrayLike, h_prev: ArrayLike, c_cur: ArrayLike) -> ArrayLike:
            return sigmoid(self.wo @ jnp.concatenate([c_cur, h_prev, x_cur], axis=0) + self.bo)
        return o_cur_jit(x_cur, h_prev, c_cur)

    def h_cur(self, x_cur: ArrayLike, h_prev: ArrayLike, c_prev: ArrayLike, c_t: ArrayLike) -> ArrayLike:
        @jax.jit
        def h_cur_jit(x_cur: ArrayLike, h_prev: ArrayLike, c_prev: ArrayLike, c_t: ArrayLike) -> ArrayLike:
            return jnp.dot(self.o_cur(x_cur, h_prev, c_t), jnp.tanh(self.c_cur(x_cur, h_prev, c_prev))) 
        return h_cur_jit(x_cur, h_prev, c_prev, c_t)

    def forward(self, x_cur: ArrayLike, h_prev: ArrayLike, c_prev: ArrayLike) -> tuple[ArrayLike, ArrayLike]:
        @jax.jit
        def forward_jit(x_cur: ArrayLike, h_prev: ArrayLike, c_prev: ArrayLike) -> tuple[ArrayLike, ArrayLike]:
            c_t = self.c_cur(x_cur, h_prev, c_prev)
            return (c_t, self.h_cur(x_cur, h_prev, c_prev, c_t))
        return forward_jit(x_cur, h_prev, c_prev)

class LSTMModel:
    def __init__(
        self,
        num_layers: int,
        lstm_type: str,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        seed: int = RANDOM_SEED
    ):
        self.num_layers = num_layers
        self.layers = []
        self.seed = seed
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        if (lstm_type == "vanilla"):
            self.layers = [LSTM(
                seed=self.seed, 
                input_dim=self.input_dim, 
                hidden_dim=self.hidden_dim) 
                for _ in range(self.num_layers)]
        elif (lstm_type == "peephole"):
            self.layers = [PeepholeLSTM(
                seed=self.seed,
                input_dim=self.input_dim, 
                hidden_dim=self.hidden_dim)
                for _ in range(self.num_layers)]
        else:
            raise ValueError("lstm_type must be 'vanilla' or 'peephole'.")
        self.key = jax.random.PRNGKey(seed=self.seed)
        self.w_out = rng_unif(key=self.key, shape=(self.hidden_dim, self.output_dim))
        self.b_out = rng_unif(key=self.key, shape=(self.hidden_dim,))
    
    def forward(self, x: ArrayLike) -> ArrayLike:
        pass
