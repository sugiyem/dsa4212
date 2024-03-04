import jax
import jax.numpy as jnp
from typing import NamedTuple
from jax import Array

RANDOM_SEED = 42

@jax.jit
def sigmoid(x: jnp.ndarray) -> jnp.ndarray:
    return 1./(1 + jnp.exp(-x))

def rng_unif(key: Array, shape: tuple[int, int]) -> jnp.ndarray:
    return jax.random.uniform(key=key, shape=shape)

class LSTMParams(NamedTuple):
    key: Array
    input_dim: int
    hidden_dim: int
    wf: jnp.ndarray
    bf: jnp.ndarray
    wi: jnp.ndarray
    bi: jnp.ndarray
    wc: jnp.ndarray
    bc: jnp.ndarray
    wo: jnp.ndarray
    bo: jnp.ndarray