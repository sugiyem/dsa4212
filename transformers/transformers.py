import jax
import jax.numpy as jnp
from utils import softmax

class Attention:
    @staticmethod
    @jax.jit
    def calc_attention(
        query: jnp.ndarray,
        key: jnp.ndarray,
        value: jnp.ndarray
    ) -> jnp.ndarray:
        dk = query.shape[0]
        return softmax((query @ key.T)/jnp.sqrt(dk)) @ value

    @staticmethod
    @jax.jit
    def calc_masked_attention(
        query: jnp.ndarray,
        key: jnp.ndarray,
        value: jnp.ndarray
    ) -> jnp.ndarray:
        pass

    @staticmethod
    @jax.jit
    def calc_multi_head_attention(
        query: jnp.ndarray,
        key: jnp.ndarray,
        value: jnp.ndarray
    )
