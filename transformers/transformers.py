import jax
import jax.numpy as jnp
from utils import InputParams, MultiHeadAttnParams, softmax, rng_unif, relu, basic_normalize

class Attention:
    @jax.jit 
    def __init__(
        self, 
        key: jax.Array, 
        input_dim: int,
        num_attention_layers: int,
        dk: int,
        dv: int
    ):
        self.key = key 
        self.input_dim = input_dim
        self.num_attention_layers = num_attention_layers
        self.dk = dk
        self.dv = dv
        self.wq = rng_unif(key=key, shape=(num_attention_layers, input_dim, dk))
        self.wk = rng_unif(key=key, shape=(num_attention_layers, input_dim, dk))
        self.wv = rng_unif(key=key, shape=(num_attention_layers, input_dim, dv))
        self.wo = rng_unif(key=key, shape=(num_attention_layers * dv, input_dim)) # output_dim = input_dim

    @staticmethod
    @jax.jit
    def calc_attention(
        query: jnp.ndarray,
        key: jnp.ndarray,
        value: jnp.ndarray,
    ) -> jnp.ndarray:
        dk = query.shape[0]
        return softmax((query @ key.T)/jnp.sqrt(dk)) @ value

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
    
    @jax.jit
    def calc_multi_head_attention(
        self,
        query: jnp.ndarray,
        key: jnp.ndarray,
        value: jnp.ndarray,
        mask: jnp.ndarray = None
    ) -> jnp.ndarray:
        concat_attention = jnp.array([])

        for i in range(self.num_attention_layers):
            q_i = query @ self.wq[i]
            k_i = key @ self.wk[i]
            v_i = value @ self.wv[i]

            scaled_attention = self.calc_attention(q_i, k_i, v_i) if mask is None \
                else Attention.calc_masked_attention(q_i, k_i, v_i, mask)
            concat_attention = jnp.append(concat_attention, scaled_attention)
        
        return concat_attention @ self.wo

# MLP with 1 hidden layer and ReLU as it's activation function
class FeedForwardNetwork:
    def __init__(
        self,
        key: jax.Array,
        input_dim: int,
        hidden_dim: int 
    ):
        self.layer1_matrix = rng_unif(key=key, shape=(input_dim, hidden_dim))
        self.layer2_matrix = rng_unif(key=key, shape=(hidden_dim, input_dim))

    def forward(self, x: jnp.ndarray) -> jnp.ndarray:
        y = x @ self.layer1_matrix
        y = relu(y)
        y = y @ self.layer2_matrix 
        return y

# Single encoder will have one multi-head attention and one feed forward NN
class SingleEncoder:
    def __init__(
        self, 
        key: jax.Array,
        input_dim: int,
        hidden_dim: int,
        num_attention_layers: int,
        dk: int,
        dv: int
    ):
        self.attention = Attention(key, input_dim, num_attention_layers, dk, dv)
        self.network = FeedForwardNetwork(key, input_dim, hidden_dim)
        
    def encode(self, x: jnp.ndarray, mask: jnp.ndarray = None) -> jnp.ndarray:
        # Use the multi head attention
        att_val = self.attention.calc_multi_head_attention(x, x, x, mask)
        # Add + normalizate
        att_val = basic_normalize(x + att_val)

        # Use the feed forward NN
        y = self.network.forward(att_val)
        # Add + normalize
        y = basic_normalize(att_val + y)

        return y
    
class Encoder:
    def __init__(
        self,
        key: jax.Array,
        input_dim: int,
        hidden_dim: int,
        num_attention_layers: int,
        dk: int,
        dv: int,
        num_encoder: int
    ):
        self.layers = [SingleEncoder(key, input_dim, hidden_dim, num_attention_layers, dk, dv) for _ in range(num_encoder)]

    def encode(self, x: jnp.ndarray, mask: jnp.ndarray = None) -> jnp.ndarray:
        y = x
        for layer in self.layers:
            y = layer.encode(y, mask)
        return y

# Single decoder will have two multi-head attentions and one feed forward NN
class SingleDecoder:
    def __init__(
        self,
        key: jax.Array,
        input_dim: int,
        hidden_dim: int,
        num_attention_layers: int,
        dk: int,
        dv: int
    ):
        self.first_attention = Attention(key, input_dim, num_attention_layers, dk, dv)
        self.second_attention = Attention(key, input_dim, num_attention_layers, dk, dv)
        self.network = FeedForwardNetwork(key, input_dim, hidden_dim)

    def decode(self,
        x: jnp.ndarray,
        encoding_mem: jnp.ndarray, # memory during encoding process
        mask1: jnp.ndarray = None,
        mask2: jnp.ndarray = None
    ) -> jnp.ndarray:
        # Use the first multi-head attention
        att_val1 = self.first_attention.calc_multi_head_attention(x, x, x, mask1)
        # Add + normalize
        att_val1 = basic_normalize(x + att_val1)

        # Use the second multi-head attention
        att_val2 = self.second_attention.calc_multi_head_attention(att_val1, encoding_mem, encoding_mem, mask2)
        # Add + normalize
        att_val2 = basic_normalize(att_val1 + att_val2)

        # Use the feed forward NN
        y = self.network.forward(y)
        # Add + normalize
        y = basic_normalize(att_val2 + y)

        return y

class Decoder:
    def __init__(
        self,
        key: jax.Array,
        input_dim: int,
        hidden_dim: int,
        num_attention_layers: int,
        dk: int,
        dv: int,
        num_decoder: int
    ):
        self.layers = [SingleDecoder(key, input_dim, hidden_dim, num_attention_layers, dk, dv) for _ in range(num_decoder)]

    def decode(self, 
        x: jnp.ndarray, 
        encoding_mem: jnp.ndarray,
        mask1: jnp.ndarray = None,
        mask2: jnp.ndarray = None
    ) -> jnp.ndarray:
        y = x
        for layer in self.layers:
            y = layer.decode(y, encoding_mem, mask1, mask2)
        return y
    
class Transformer:
    def __init__(
        self,
        key: jax.Array,
        input_dim: int,
        hidden_dim: int,
        num_attention_layers: int,
        dk: int,
        dv: int,
        num_encoder: int,
        num_decoder: int
    ):
        self.encoder = Encoder(key, input_dim, hidden_dim, num_attention_layers, dk, dv, num_encoder)
        self.decoder = Decoder(key, input_dim, hidden_dim, num_attention_layers, dk, dv, num_decoder)

    def forward(self,
        input: jnp.ndarray,
        output: jnp.ndarray,
        input_mask: jnp.ndarray = None,
        output_mask: jnp.ndarray = None
    ) -> jnp.ndarray:
        encoding_mem = self.encoder.forward(input, input_mask)
        return self.decoder.decode(output, encoding_mem, input_mask, output_mask)