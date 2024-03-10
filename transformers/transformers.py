import jax
import jax.numpy as jnp
from utils import InputParams, MultiHeadAttnParams, softmax, rng_unif

class Attention:
    @staticmethod
    @jax.jit
    def init_input_params(
        key: jax.Array,
        input_dim: int, 
        dk: int, 
        dv: int
    ) -> InputParams:
        wq = rng_unif(key=key, shape=(input_dim, dk))
        wk = rng_unif(key=key, shape=(input_dim, dk))
        wv = rng_unif(key=key, shape=(input_dim, dv))
        return InputParams(wq, wk, wv)

    @staticmethod
    @jax.jit
    def calc_attention(
        params: InputParams,
        query: jnp.ndarray,
        key: jnp.ndarray,
        value: jnp.ndarray
    ) -> jnp.ndarray:
        dk = query.shape[0]
        return softmax((query @ key.T)/jnp.sqrt(dk)) @ value

    @staticmethod
    @jax.jit
    def calc_masked_attention(
        params: InputParams,
        query: jnp.ndarray,
        key: jnp.ndarray,
        value: jnp.ndarray
    ) -> jnp.ndarray:
        pass

    @staticmethod
    @jax.jit
    def calc_multi_head_attention(
        params: MultiHeadAttnParams,
        query: jnp.ndarray,
        key: jnp.ndarray,
        value: jnp.ndarray
    ) -> jnp.ndarray:
        pass
