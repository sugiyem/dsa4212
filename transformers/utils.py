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

# create a mask of shape (1, size, size)
def generate_mask(size: int) -> jnp.ndarray:
    mask = jnp.triu(jnp.ones((1, size, size), dtype=int), k=1)
    return mask == 0

class Batch:
    "Object for holding a batch of data with mask during training."
    "Source: https://nlp.seas.harvard.edu/2018/04/03/attention.html"

    def __init__(self, 
        src: jnp.ndarray, # must be a 2D jnp.ndarray
        tgt: jnp.ndarray, # must be a 2D jnp.ndarray
        pad: int = 0
    ):
        self.src = src 
        self.src_mask = (src != pad).reshape(src.shape[0], 1, src.shape[1])
        self.tgt = tgt[:, :-1]
        self.tgt_final = tgt[:, 1:]
        self.tgt_mask = self.make_std_mask(self.tgt, pad)
    
    @staticmethod
    def make_std_mask(tgt: jnp.ndarray, pad: jnp.ndarray) -> jnp.ndarray:
        "Create a mask to hide padding and future words."
        dim_x, dim_y = tgt.shape
        tgt_mask = (tgt != pad).reshape(dim_x, 1, dim_y)
        tgt_mask = tgt_mask & generate_mask(dim_y)
        return tgt_mask
    
def create_train_state(
    model: nn.Module,
    learning_rate: float, 
    key: jax.Array
) -> train_state.TrainState:
    dummy_array = jnp.ones((1, 1), dtype=int)
    dummy_mask = jnp.ones((1, 1, 1), dtype=int)
    params = model.init(key, dummy_array, dummy_array, dummy_mask, dummy_mask)['params']
    tx = optax.adam(learning_rate)

    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

# Train the state
def train_step(state: train_state.TrainState, train_data: Batch) -> tuple[float, train_state.TrainState]:
    def loss_fn(params):
        logits = state.apply_fn({'params': params}, train_data.src, train_data.tgt, train_data.src_mask, train_data.tgt_mask)
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits, labels=train_data.tgt_final
        ).mean()
        return loss 
    
    loss_grad_fn = jax.value_and_grad(loss_fn)
    loss, grad = loss_grad_fn(state.params)
    state = state.apply_gradients(grads=grad)

    return loss, state

def train_model(state: train_state.TrainState, data_generator: callable, num_epoch: int) -> train_state.TrainState:
    for i in range(num_epoch):
        total_loss = 0.
        count = 0
        train_loader = data_generator()

        for train_batch in train_loader:
            loss, state = train_step(state, train_batch)
            total_loss += loss 
            count += 1

        avg_loss = total_loss / count
        print("Epoch: {}, Loss: {}".format(i, avg_loss))

    return state

# the decode doesn't use any masking currently
# need to fix this later
def decode(
    state: train_state.TrainState, 
    input: jnp.ndarray, # must be of size (num_test_case, input_seq_len)
    output_init: jnp.ndarray, # must be of size (num_test_case, 1)
    output_len: int,
    input_mask: jnp.ndarray = None
) -> jnp.ndarray:
    '''
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len-1):
        out = model.decode(memory, src_mask, 
                           Variable(ys), 
                           Variable(subsequent_mask(ys.size(1))
                                    .type_as(src.data)))
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim = 1)
        next_word = next_word.data[0]
        ys = torch.cat([ys, 
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return ys
    '''
    curr_output = output_init
    for _ in range(output_len - 1):
        logits = state.apply_fn({'params': state.params}, input, curr_output, input_mask, generate_mask(curr_output.shape[1]))
        
        # only consider output for last sequence
        last_logits = logits[:, -1, :]
        # greedy decoding: pick the word with largest probability
        next_word = jnp.argmax(last_logits, axis=1)

        curr_output = jnp.hstack((curr_output, next_word.reshape(-1, 1)))
    
    return curr_output