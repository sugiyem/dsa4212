import jax
import jax.numpy as jnp
import flax 
import optax
from flax.training import train_state
from flax import linen as nn
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

@jax.jit 
def relu(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.maximum(x, 0)

@jax.jit
def basic_normalize(x: jnp.ndarray) -> jnp.ndarray:
    return (x - jnp.mean(x)) / (jnp.std(x) + 1e-6)

class Batch:
    "Object for holding a batch of data with mask during training."
    "Source: https://nlp.seas.harvard.edu/2018/04/03/attention.html"

    def __init__(self, 
        src: jnp.ndarray, 
        tgt: jnp.ndarray, 
        pad: int = 0
    ):
        self.src = src 
        self.src_mask = None 
        self.tgt = tgt[:, :-1] # pad the target
        self.tgt_final = tgt[:, 1:]
        self.tgt_mask = None

        # Need to fix the masking later
        '''
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if trg is not None:
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]
            self.trg_mask = \
                self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()

        '''
    
    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        '''
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask
        '''
        pass
    
def create_train_state(
    model: nn.Module,
    learning_rate: float, 
    key: jax.Array,
    batch_size: int,
    input_seq_len: int,
    output_seq_len: int
) -> train_state.TrainState:
    params = model.init(key, jnp.ones((1,1), dtype=int), jnp.ones((1,1), dtype=int), None, None)['params']
    tx = optax.adam(learning_rate)

    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

# Train the state
def train_step(state: train_state.TrainState, train_data: Batch) -> train_state.TrainState:
    def loss_fn(params):
        logits = state.apply_fn({'params': params}, train_data.src, train_data.tgt, train_data.src_mask, train_data.tgt_mask)
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits, labels=train_data.tgt_final
        ).mean()
        return loss 
    
    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)

    return state

def train_model(state: train_state.TrainState, data_generator: callable, num_epoch: int) -> train_state.TrainState:
    for _ in range(num_epoch):
        train_loader = data_generator()
        for train_batch in train_loader:
            state = train_step(state, train_batch)

    return state

# the decode doesn't use any masking currently
# need to fix this later
def decode(
    state: train_state.TrainState, 
    input: jnp.ndarray, # must be of size (num_test_case, input_seq_len)
    output_init: jnp.ndarray, # must be of size (num_test_case, 1)
    output_len: int
) -> jnp.ndarray:
    curr_output = output_init
    for _ in range(output_len - 1):
        print(curr_output)
        logits = state.apply_fn({'params': state.params}, input, curr_output, None, None)
        
        # only consider output for last sequence
        last_logits = logits[:, -1, :]
        # greedy decoding: pick the word with largest probability
        next_word = jnp.argmax(last_logits, axis=1)

        curr_output = jnp.hstack((curr_output, next_word.reshape(-1, 1)))
    
    return curr_output