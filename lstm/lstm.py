import jax
import jax.numpy as jnp
from utils import RANDOM_SEED, rng_normal, sigmoid, LSTMParams, LSTMArchiParams, rng_unif

class LSTM:
    @staticmethod
    def init_params(
        seed: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: int
    ) -> tuple[LSTMArchiParams, LSTMParams]:
        key = jax.random.PRNGKey(seed)
        wf = rng_unif(key=key, shape=(hidden_dim, input_dim))
        uf = rng_unif(key=key, shape=(hidden_dim, hidden_dim))
        bf = rng_unif(key=key, shape=(hidden_dim, 1))
        wi = rng_unif(key=key, shape=(hidden_dim, input_dim))
        ui = rng_unif(key=key, shape=(hidden_dim, hidden_dim))
        bi = rng_unif(key=key, shape=(hidden_dim, 1))
        wc = rng_unif(key=key, shape=(hidden_dim, input_dim))
        uc = rng_unif(key=key, shape=(hidden_dim, hidden_dim))
        bc = rng_unif(key=key, shape=(hidden_dim, 1))
        wo = rng_unif(key=key, shape=(hidden_dim, input_dim))
        uo = rng_unif(key=key, shape=(hidden_dim, hidden_dim))
        bo = rng_unif(key=key, shape=(hidden_dim, 1))
        wout = rng_unif(key=key, shape=(output_dim, hidden_dim))
        return LSTMArchiParams(key, input_dim, hidden_dim, output_dim), \
            LSTMParams(wf, uf, bf, wi, ui, bi, wc, uc, bc, wo, uo, bo, wout)

    @staticmethod
    @jax.jit
    def f_cur(
        params: LSTMParams, 
        x_cur: jnp.ndarray, 
        h_prev: jnp.ndarray
    ) -> jnp.ndarray:
        return sigmoid((params.uf @ h_prev) + (params.wf @ x_cur) + params.bf)
    
    @staticmethod
    @jax.jit
    def i_cur(
        params: LSTMParams, 
        x_cur: jnp.ndarray, 
        h_prev: jnp.ndarray
    ) -> jnp.ndarray:
        return sigmoid((params.ui @ h_prev) + (params.wi @ x_cur) + params.bi)

    @staticmethod
    @jax.jit
    def c_cur_hat(
        params: LSTMParams,
        x_cur: jnp.ndarray,
        h_prev: jnp.ndarray
    ) -> jnp.ndarray:
        return jnp.tanh((params.uc @ h_prev) + (params.wc @ x_cur) + params.bc)

    @staticmethod
    @jax.jit
    def c_cur(
        params: LSTMParams, 
        x_cur: jnp.ndarray, 
        h_prev: jnp.ndarray, 
        c_prev: jnp.ndarray
    ) -> jnp.ndarray:
        i_t = LSTM.i_cur(params, x_cur, h_prev)
        c_t_hat = LSTM.c_cur_hat(params, x_cur, h_prev)
        f_t = LSTM.f_cur(params, x_cur, h_prev)
        return f_t * c_prev + i_t * c_t_hat
        
    
    @staticmethod
    @jax.jit
    def o_cur(
        params: LSTMParams, 
        x_cur: jnp.ndarray, 
        h_prev: jnp.ndarray
    ) -> jnp.ndarray:
        return sigmoid(params.uo @ h_prev + params.wo @ x_cur + params.bo)
    
    @staticmethod
    @jax.jit
    def h_cur(
        params: LSTMParams, 
        x_cur: jnp.ndarray, 
        h_prev: jnp.ndarray, 
        c_prev: jnp.ndarray
    ) -> jnp.ndarray:
        o_t = LSTM.o_cur(params, x_cur, h_prev)
        c_t = LSTM.c_cur(params, x_cur, h_prev, c_prev)
        return o_t * jnp.tanh(c_t), c_t
        
    @staticmethod
    def forward(
        tup: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
        x_cur: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        params, h_prev, c_prev = tup
        h_t, c_t = LSTM.h_cur(params, x_cur, h_prev, c_prev)
        out_t = params.wout @ h_t
        return (params, h_t, c_t), out_t

    @staticmethod
    def forward_full(
        archi_params: LSTMArchiParams,
        params: LSTMParams, 
        x_in: jnp.ndarray
    ) -> jnp.ndarray:
        h, c = jnp.zeros(shape=(archi_params.hidden_dim, 1)), jnp.zeros(shape=(archi_params.hidden_dim, 1))
        (params, h, c), out_ls = jax.lax.scan(LSTM.forward, (params, h, c), x_in[0])
        return out_ls
    
    @staticmethod
    def forward_batch(
        archi_params: LSTMArchiParams,
        params: LSTMParams,
        x_batch: jnp.ndarray
    ) -> jnp.ndarray:
        forward_full_batch = jax.vmap(LSTM.forward_full, in_axes=(None, None, 0))
        return jnp.squeeze(forward_full_batch(archi_params, params, x_batch), axis=3)
    
    @staticmethod
    def mse(
        archi_params: LSTMArchiParams,
        params: LSTMParams,
        x_batch: jnp.ndarray,
        y_batch: jnp.ndarray
    ) -> jnp.ndarray:
        batch_out = LSTM.forward_batch(archi_params, params, x_batch)
        return jnp.mean((batch_out - y_batch) ** 2)
    
    @staticmethod
    def backward(
        archi_params: LSTMArchiParams,
        params: LSTMParams,
        x_batch: jnp.ndarray,
        y_batch: jnp.ndarray
    ) -> jnp.ndarray:
        mse_grad = jax.jacfwd(LSTM.mse, argnums=(1,))
        cur_grad = mse_grad(archi_params, params, x_batch, y_batch)
        return cur_grad

