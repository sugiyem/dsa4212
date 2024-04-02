import jax
import jax.numpy as jnp
from typing import NamedTuple

class InputParams(NamedTuple):
    wq: jnp.ndarray
    wk: jnp.ndarray
    wv: jnp.ndarray

class SingleHeadAttnParams(NamedTuple):
    wq_i: jnp.ndarray
    wk_i: jnp.ndarray
    wv_i: jnp.ndarray

class MultiHeadAttnParams(NamedTuple):
    w_full: list[SingleHeadAttnParams]
    wo: jnp.ndarray

@jax.jit
def rng_unif(key: jax.Array, shape) -> jnp.ndarray:
    return jax.random.uniform(key=key, shape=shape)

@jax.jit
def softmax(x: jnp.ndarray) -> jnp.ndarray:
    x_exp = jnp.exp(x)
    return x_exp / jnp.sum(x_exp)

@jax.jit
def calc_inputs(params: InputParams, x: jnp.ndarray) -> jnp.ndarray:
    q = x @ params.wq
    k = x @ params.wk
    v = x @ params.wv
    return jnp.array([q, k, v])
