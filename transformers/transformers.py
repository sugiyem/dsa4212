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
        query: jnp.ndarray,
        key: jnp.ndarray,
        value: jnp.ndarray,
    ) -> jnp.ndarray:
        dk = query.shape[0]
        return softmax((query @ key.T)/jnp.sqrt(dk)) @ value

    @staticmethod
    @jax.jit
    def calc_masked_attention(
        query: jnp.ndarray,
        key: jnp.ndarray,
        value: jnp.ndarray,
        mask: jnp.ndarray
    ) -> jnp.ndarray:
        dk = query.shape[0]
        scores = (query @ key.T) / jnp.sqrt(dk)
        masked_scores = jnp.where(mask == 0, -jnp.inf, scores)
        return softmax(masked_scores) @ value
    
    @staticmethod
    @jax.jit
    def calc_multi_head_attention(
        params: MultiHeadAttnParams,
        query: jnp.ndarray,
        key: jnp.ndarray,
        value: jnp.ndarray,
        mask: jnp.ndarray = None
    ) -> jnp.ndarray:
        concat_attention = jnp.array([])

        for attention_param in params.w_full:
            wq_i, wk_i, wv_i = attention_param
            q_i = query @ wq_i 
            k_i = key @ wk_i 
            v_i = value @ wv_i

            scaled_attention = Attention.calc_attention(q_i, k_i, v_i) if mask is None \
                else Attention.calc_masked_attention(q_i, k_i, v_i, mask)
            concat_attention = jnp.append(concat_attention, scaled_attention)
        
        return concat_attention @ params.wo

