import jax
import jax.numpy as jnp
from typing import NamedTuple
from jax import Array

RANDOM_SEED = 42

@jax.jit
def sigmoid(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.where(x >= 0, 1./(1. + jnp.exp(-x)), jnp.exp(x)/(1. + jnp.exp(x)))

def rng_unif(key: Array, shape: tuple[int, int]) -> jnp.ndarray:
    return jax.random.uniform(key=key, shape=shape, minval=-1/jnp.sqrt(shape[0]), maxval=1/jnp.sqrt(shape[0]))

def rng_normal(key: Array, shape: tuple[int, int]) -> jnp.ndarray:
    return jax.random.normal(key=key, shape=shape) / jnp.sqrt(shape[0])

@jax.jit
def mse(y_pred: jnp.ndarray, y_true: jnp.ndarray) -> jnp.ndarray:
    return jnp.mean((y_pred - y_true) ** 2)

class LSTMParams(NamedTuple):
    wf: jnp.ndarray
    uf: jnp.ndarray
    bf: jnp.ndarray
    wi: jnp.ndarray
    ui: jnp.ndarray
    bi: jnp.ndarray
    wc: jnp.ndarray
    uc: jnp.ndarray
    bc: jnp.ndarray
    wo: jnp.ndarray
    uo: jnp.ndarray
    bo: jnp.ndarray
    wout: jnp.ndarray

class LSTMArchiParams(NamedTuple):
    key: Array
    input_dim: int
    hidden_dim: int
    output_dim: int

