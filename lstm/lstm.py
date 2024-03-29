import jax
import jax.numpy as jnp
from typing import Callable
from utils import RANDOM_SEED, rng_normal, sigmoid, LSTMParams, \
    LSTMModelParams

class LSTM:
    @staticmethod
    def init_params(
        seed: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: int
    ) -> LSTMParams:
        key = jax.random.PRNGKey(seed)
        wf = rng_normal(key=key, shape=(hidden_dim, hidden_dim + input_dim))
        bf = rng_normal(key=key, shape=(hidden_dim,))
        wi = rng_normal(key=key, shape=(hidden_dim, hidden_dim + input_dim))
        bi = rng_normal(key=key, shape=(hidden_dim,))
        wc = rng_normal(key=key, shape=(hidden_dim, hidden_dim + input_dim))
        bc = rng_normal(key=key, shape=(hidden_dim,))
        wo = rng_normal(key=key, shape=(hidden_dim, hidden_dim + input_dim))
        bo = rng_normal(key=key, shape=(hidden_dim,))
        wout = rng_normal(key=key, shape=(output_dim, hidden_dim))
        return LSTMParams(key, input_dim, hidden_dim, output_dim, wf, bf, wi, bi, wc, bc, wo, bo, wout)

    @staticmethod
    @jax.jit
    def f_cur(
        params: LSTMParams, 
        x_cur: jnp.ndarray, 
        h_prev: jnp.ndarray
    ) -> jnp.ndarray:
        return sigmoid((params.wf @ jnp.concatenate([h_prev, x_cur], axis=0)) + params.bf)
    
    @staticmethod
    @jax.jit
    def i_cur(
        params: LSTMParams, 
        x_cur: jnp.ndarray, 
        h_prev: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        i_t = sigmoid((params.wi @ jnp.concatenate([h_prev, x_cur], axis=0)) + params.bi)
        c_t_hat = jnp.tanh((params.wc @ jnp.concatenate([h_prev, x_cur], axis=0)) + params.bc)
        return i_t, c_t_hat

    @staticmethod
    @jax.jit
    def c_cur(
        params: LSTMParams, 
        x_cur: jnp.ndarray, 
        h_prev: jnp.ndarray, 
        c_prev: jnp.ndarray
    ) -> jnp.ndarray:
        i_t, c_t_hat = LSTM.i_cur(params, x_cur, h_prev)
        return jnp.dot(LSTM.f_cur(params, x_cur, h_prev), c_prev) + jnp.dot(i_t, c_t_hat)
        
    
    @staticmethod
    @jax.jit
    def o_cur(
        params: LSTMParams, 
        x_cur: jnp.ndarray, 
        h_prev: jnp.ndarray
    ) -> jnp.ndarray:
        return sigmoid((params.wo @ jnp.concatenate([h_prev, x_cur], axis=0)) + params.bo)
    
    @staticmethod
    @jax.jit
    def h_cur(
        params: LSTMParams, 
        x_cur: jnp.ndarray, 
        h_prev: jnp.ndarray, 
        c_prev: jnp.ndarray
    ) -> jnp.ndarray:
        return jnp.dot(LSTM.o_cur(params, x_cur, h_prev), 
                       jnp.tanh(LSTM.c_cur(params, x_cur, h_prev, c_prev)))
        
    @staticmethod
    @jax.jit
    def forward(
        params: LSTMParams,
        x_cur: jnp.ndarray, 
        h_prev: jnp.ndarray, 
        c_prev: jnp.ndarray,
        wout: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        h_t = LSTM.h_cur(params, x_cur, h_prev, c_prev)
        o_t = wout @ h_t
        return (LSTM.c_cur(params, x_cur, h_prev, c_prev), o_t, h_t)

    @staticmethod
    @jax.jit
    def forward_full(
        params: LSTMParams, 
        x_in: jnp.ndarray
    ) -> jnp.ndarray:
        time_steps = x_in.shape[0]
        h, o, c = jnp.zeros(shape=(params.hidden_dim,)), jnp.zeros(shape=(params.output_dim,)), \
            jnp.zeros(shape=(params.hidden_dim,))
        o_ls = []
        for i in range(len(time_steps)):
            c_t, o_t, h_t = LSTM.forward(params, x_in[i,:], h, c, params.wout)
            o_ls.append(o_t)
            h, o, c = h_t, o_t, c_t

        return jnp.array(o_ls)

class PeepholeLSTM(LSTM):
    @staticmethod
    @jax.jit
    def f_cur(
        params: LSTMParams, 
        x_cur: jnp.ndarray, 
        h_prev: jnp.ndarray, 
        c_prev: jnp.ndarray
    ) -> jnp.ndarray:
        return sigmoid(params.wf @ jnp.concatenate([c_prev, h_prev, x_cur], axis=0) + params.bf)

    @staticmethod
    @jax.jit
    def i_cur(
        params: LSTMParams, 
        x_cur: jnp.ndarray, 
        h_prev: jnp.ndarray, 
        c_prev: jnp.ndarray
    ) -> jnp.ndarray:
        return sigmoid(params.wi @ jnp.concatenate([c_prev, h_prev, x_cur], axis=0) + params.bi)
    
    @staticmethod
    @jax.jit
    def o_cur(
        params: LSTMParams, 
        x_cur: jnp.ndarray, 
        h_prev: jnp.ndarray, 
        c_cur: jnp.ndarray
    ) -> jnp.ndarray:
        return sigmoid(params.wo @ jnp.concatenate([c_cur, h_prev, x_cur], axis=0) + params.bo)

    @staticmethod
    @jax.jit
    def h_cur(
        params: LSTMParams, 
        x_cur: jnp.ndarray, 
        h_prev: jnp.ndarray, 
        c_prev: jnp.ndarray, 
        c_t: jnp.ndarray
    ) -> jnp.ndarray:
        return jnp.dot(PeepholeLSTM.o_cur(params, x_cur, h_prev, c_t), 
                       jnp.tanh(PeepholeLSTM.c_cur(params, x_cur, h_prev, c_prev))) 

    @staticmethod
    @jax.jit
    def forward(
        params: LSTMParams,
        x_cur: jnp.ndarray, 
        h_prev: jnp.ndarray, 
        c_prev: jnp.ndarray,
        wout: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        c_t = PeepholeLSTM.c_cur(params, x_cur, h_prev, c_prev)
        h_t = PeepholeLSTM.h_cur(params, x_cur, h_prev, c_prev, c_t)
        o_t = wout @ h_t
        return (c_t, o_t, h_t)

    @staticmethod
    @jax.jit
    def forward_full(params: LSTMParams, x_in: jnp.ndarray) -> jnp.ndarray:
        time_steps = x_in.shape[0]
        h, o, c = jnp.zeros(shape=(params.hidden_dim,)), jnp.zeros(shape=(params.output_dim,)), \
            jnp.zeros(shape=(params.hidden_dim,))
        o_ls = []
        for i in range(len(time_steps)):
            c_t, o_t, h_t = PeepholeLSTM.forward(x_in[i,:], h, c, params.wout)
            o_ls.append(o_t)
            h, o, c = h_t, o_t, c_t
        return jnp.array(o_ls)

class LSTMModel:
    def init_params(
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_lstm: int,
        lstm_type: str,
        seed: int = RANDOM_SEED
    ):
        assert num_lstm >= 1, "num_lstm must be >= 1"
        if (lstm_type == "vanilla"):
            layers = [LSTM.init_params(
                seed=seed, 
                input_dim=input_dim, 
                hidden_dim=hidden_dim,
                output_dim=output_dim
            ) 
                for _ in range(num_lstm)]
        elif (lstm_type == "peephole"):
            layers = [PeepholeLSTM.init_params(
                seed=seed,
                input_dim=input_dim, 
                hidden_dim=hidden_dim,
                output_dim=output_dim
            )
                for _ in range(num_lstm)]
        else:
            raise ValueError("lstm_type must be 'vanilla' or 'peephole'.")
        return LSTMModelParams(num_lstm, lstm_type, layers)

    def forward(params: LSTMModelParams, x_in: jnp.ndarray) -> jnp.ndarray:
        num_lstm = params.num_lstm
        o_out = jnp.zeros(shape=(x_in.shape[0], params.layers[0].output_dim))
        for i in range(num_lstm):
            cur_params = params.layers[i]
            o_out = type(params.layers[0]).forward_full(cur_params, x_in)
        return o_out
        