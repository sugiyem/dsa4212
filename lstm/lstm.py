import jax
import jax.numpy as jnp
from typing import Callable
from utils import RANDOM_SEED, rng_unif, sigmoid, LSTMParams

class LSTM:
    @staticmethod
    def initialise_params(
        seed: int,
        input_dim: int,
        hidden_dim: int
    ) -> LSTMParams:
        key = jax.random.PRNGKey(seed)
        wf = rng_unif(key=key, shape=(hidden_dim, hidden_dim + input_dim))
        bf = rng_unif(key=key, shape=(hidden_dim,))
        wi = rng_unif(key=key, shape=(hidden_dim, hidden_dim + input_dim))
        bi = rng_unif(key=key, shape=(hidden_dim,))
        wc = rng_unif(key=key, shape=(hidden_dim, hidden_dim + input_dim))
        bc = rng_unif(key=key, shape=(hidden_dim,))
        wo = rng_unif(key=key, shape=(hidden_dim, hidden_dim + input_dim))
        bo = rng_unif(key=key, shape=(hidden_dim,))
        return LSTMParams(key, input_dim, hidden_dim, wf, bf, wi, bi, wc, bc, wo, bo)

    @staticmethod
    @jax.jit
    def f_cur(params: LSTMParams, x_cur: jnp.ndarray, h_prev: jnp.ndarray) -> jnp.ndarray:
        return sigmoid((params.wf @ jnp.concatenate([h_prev, x_cur], axis=0)) + params.bf)
    
    @staticmethod
    @jax.jit
    def i_cur(params: LSTMParams, x_cur: jnp.ndarray, h_prev: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        i_t = sigmoid((params.wi @ jnp.concatenate([h_prev, x_cur], axis=0)) + params.bi)
        c_t_hat = jnp.tanh((params.wc @ jnp.concatenate([h_prev, x_cur], axis=0)) + params.bc)
        return i_t, c_t_hat

    @staticmethod
    @jax.jit
    def c_cur(params: LSTMParams, x_cur: jnp.ndarray, h_prev: jnp.ndarray, c_prev: jnp.ndarray) -> jnp.ndarray:
        i_t, c_t_hat = params.i_cur(x_cur, h_prev)
        return jnp.dot(params.f_cur(x_cur, h_prev), c_prev) + jnp.dot(i_t, c_t_hat)
        
    
    @staticmethod
    @jax.jit
    def o_cur(params: LSTMParams, x_cur: jnp.ndarray, h_prev: jnp.ndarray) -> jnp.ndarray:
        return sigmoid((params.wo @ jnp.concatenate([h_prev, x_cur], axis=0)) + params.bo)
    
    @staticmethod
    @jax.jit
    def h_cur(params: LSTMParams, x_cur: jnp.ndarray, h_prev: jnp.ndarray, c_prev: jnp.ndarray) -> jnp.ndarray:
        return jnp.dot(params.o_cur(x_cur, h_prev), jnp.tanh(params.c_cur(x_cur, h_prev, c_prev)))
        
    @staticmethod
    @jax.jit
    def forward(params: LSTMParams, x_cur: jnp.ndarray, h_prev: jnp.ndarray, c_prev: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        return (params.c_cur(x_cur, h_prev, c_prev), params.h_cur(x_cur, h_prev, c_prev))

    @staticmethod
    @jax.jit
    def forward_full(params: LSTMParams, x_in: jnp.ndarray) -> jnp.ndarray:
        time_steps = x_in.shape[0]
        h, c = jnp.zeros(shape=(params.hidden_dim,)), jnp.zeros(shape=(params.hidden_dim,))
        h_ls = []
        for i in range(len(time_steps)):
            c_t, h_t = params.forward(x_in[i,:], h, c)
            h_ls.append(h_t)
            h, c = h_t, c_t
        return jnp.array(h_ls)
    
    @staticmethod
    @jax.jit
    def backward_full(params: LSTMParams, h_out: jnp.ndarray, y_true: jnp.ndarray) -> jnp.ndarray:
        pass
            



class PeepholeLSTM(LSTM):
    
    @staticmethod
    @jax.jit
    def f_cur(params: LSTMParams, x_cur: jnp.ndarray, h_prev: jnp.ndarray, c_prev: jnp.ndarray) -> jnp.ndarray:
        return sigmoid(params.wf @ jnp.concatenate([c_prev, h_prev, x_cur], axis=0) + params.bf)

    @staticmethod
    @jax.jit
    def i_cur(params: LSTMParams, x_cur: jnp.ndarray, h_prev: jnp.ndarray, c_prev: jnp.ndarray) -> jnp.ndarray:
        return sigmoid(params.wi @ jnp.concatenate([c_prev, h_prev, x_cur], axis=0) + params.bi)
    
    @staticmethod
    @jax.jit
    def o_cur(params: LSTMParams, x_cur: jnp.ndarray, h_prev: jnp.ndarray, c_cur: jnp.ndarray) -> jnp.ndarray:
        return sigmoid(params.wo @ jnp.concatenate([c_cur, h_prev, x_cur], axis=0) + params.bo)

    @staticmethod
    @jax.jit
    def h_cur(params: LSTMParams, x_cur: jnp.ndarray, h_prev: jnp.ndarray, c_prev: jnp.ndarray, c_t: jnp.ndarray) -> jnp.ndarray:
        return jnp.dot(params.o_cur(x_cur, h_prev, c_t), jnp.tanh(params.c_cur(x_cur, h_prev, c_prev))) 

    @staticmethod
    @jax.jit
    def forward(params: LSTMParams, x_cur: jnp.ndarray, h_prev: jnp.ndarray, c_prev: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        c_t = params.c_cur(x_cur, h_prev, c_prev)
        return (c_t, params.h_cur(x_cur, h_prev, c_prev, c_t))

    @staticmethod
    @jax.jit
    def backward(params, f: Callable, h_cur: jnp.ndarray, y_cur: jnp.ndarray) -> float:
        cur_err = f(h_cur, y_cur)
        pass

class LSTMModel:
    def __init__(
        params,
        num_lstm: int,
        lstm_type: str,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        seed: int = RANDOM_SEED
    ):
        params.num_lstm = num_lstm
        params.layers = []
        params.seed = seed
        params.input_dim = input_dim
        params.output_dim = output_dim
        params.hidden_dim = hidden_dim
        if (lstm_type == "vanilla"):
            params.layers = [LSTM(
                seed=params.seed, 
                input_dim=params.input_dim, 
                hidden_dim=params.hidden_dim) 
                for _ in range(params.num_lstm)]
        elif (lstm_type == "peephole"):
            params.layers = [PeepholeLSTM(
                seed=params.seed,
                input_dim=params.input_dim, 
                hidden_dim=params.hidden_dim)
                for _ in range(params.num_lstm)]
        else:
            raise ValueError("lstm_type must be 'vanilla' or 'peephole'.")
        params.key = jax.random.PRNGKey(seed=params.seed)
        params.w_out = rng_unif(key=params.key, shape=(params.hidden_dim, params.output_dim))
        params.b_out = rng_unif(key=params.key, shape=(params.output_dim,))