import jax
import jax.numpy as jnp
import numpy as np
from utils import InputParams, MultiHeadAttnParams, softmax, rng_unif, relu, basic_normalize

class Attention:
    def __init__(
        self, 
        seed: int,
        model_dim: int,
        num_att_layers: int,
        dk: int, # key dimension
        dv: int # value dimension
    ):
        self.num_att_layers = num_att_layers
        key = jax.random.PRNGKey(seed)
        self.wq = rng_unif(key=key, shape=(num_att_layers, model_dim, dk))
        self.wk = rng_unif(key=key, shape=(num_att_layers, model_dim, dk))
        self.wv = rng_unif(key=key, shape=(num_att_layers, model_dim, dv))
        self.wo = rng_unif(key=key, shape=(num_att_layers * dv, model_dim))

    @staticmethod
    @jax.jit
    def calc_attention(
        query: jnp.ndarray,
        key: jnp.ndarray,
        value: jnp.ndarray,
    ) -> jnp.ndarray:
        dk = query.shape[1]
        return jax.nn.softmax((query @ key.T) / jnp.sqrt(dk)) @ value

    @jax.jit
    def calc_masked_attention(
        query: jnp.ndarray,
        key: jnp.ndarray,
        value: jnp.ndarray,
        mask: jnp.ndarray
    ) -> jnp.ndarray:
        dk = query.shape[1]
        scores = (query @ key.T) / jnp.sqrt(dk)
        masked_scores = jnp.where(mask == 0, -jnp.inf, scores)
        return jax.nn.softmax(masked_scores) @ value
    
    def calc_multi_head_attention(
        self,
        query: jnp.ndarray,
        key: jnp.ndarray,
        value: jnp.ndarray,
        mask: jnp.ndarray = None
    ) -> jnp.ndarray:
        # query, key, value, mask must all be a jnp.array of size (len_seq, dim_size)
        attentions = []

        for i in range(self.num_att_layers):
            q_i = query @ self.wq[i]
            k_i = key @ self.wk[i]
            v_i = value @ self.wv[i]

            # q_i and v_i must be a jnp.array of size (len_seq, dk)
            # v_i must be a jnp.array of size (len_seq, dv)

            scaled_attention = self.calc_attention(q_i, k_i, v_i) if mask is None \
                else Attention.calc_masked_attention(q_i, k_i, v_i, mask)
            attentions.append(scaled_attention)
        
        concat_attention = jnp.concatenate(attentions, axis=1)
        return concat_attention @ self.wo # the output is a jnp.array of size (len_seq, dim_size)

# MLP with 1 hidden layer and ReLU as it's activation function
class FeedForwardNetwork:
    def __init__(
        self,
        seed: int,
        model_dim: int,
        hidden_dim: int 
    ):
        key = jax.random.PRNGKey(seed)
        self.layer1_matrix = rng_unif(key=key, shape=(model_dim, hidden_dim))
        self.layer2_matrix = rng_unif(key=key, shape=(hidden_dim, model_dim))

    def forward(self, x: jnp.ndarray) -> jnp.ndarray:
        y = x @ self.layer1_matrix
        y = relu(y)
        y = y @ self.layer2_matrix 
        return y

class Embedding:
    def __init__(
        self,
        seed: int,
        num_vocab: int,
        model_dim: int
    ):
        self.num_vocab = num_vocab 
        self.model_dim = model_dim
        key=jax.random.PRNGKey(seed)
        self.embed_matrix = rng_unif(key=key, shape=(num_vocab, model_dim))

    def embed(self, x: jnp.ndarray) -> jnp.ndarray:
        # input x is a jnp.array of size seq_len
        return self.embed_matrix[x] * jnp.sqrt(self.model_dim) # output is a jnp.array of size (seq_len, model_dim)
    
class PositionalEncoder:
    def __init__(
        self,
        model_dim: int,
        max_seq_len: int = 5000
    ):
        assert model_dim % 2 == 0, "model dimension must be even"
        position = np.arange(max_seq_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, model_dim, 2) * -(np.log(10000.0) / model_dim))
        
        self.position_encoding = np.zeros((max_seq_len, model_dim))
        self.position_encoding[:, 0::2] = np.sin(position * div_term)
        self.position_encoding[:, 1::2] = np.cos(position * div_term)
        self.position_encoding = jnp.array(self.position_encoding)
        

    def encode(self, x: jnp.ndarray) -> jnp.ndarray:
        # input x is a jnp.array of size (seq_len, model_dim)
        seq_len = x.shape[0]
        return x + self.position_encoding[:seq_len, :] # output will still retain the dimension (seq_len, model_dim)

# Represents the whole pre-processing step before the real encoding steps in
# Embeddding + Positional Encoding
class Preprocessing:
    def __init__(
        self,
        seed: int,
        num_vocab: int,
        model_dim: int,
        max_seq_len: int = 5000
    ):
        self.embedding = Embedding(seed, num_vocab, model_dim)
        self.positional_encoder = PositionalEncoder(model_dim, max_seq_len)

    def preprocess(self, x: jnp.ndarray) -> jnp.ndarray:
        y = self.embedding.embed(x)
        y = self.positional_encoder.encode(y)
        return y
    
# Single encoder will have one multi-head attention and one feed forward NN
class SingleEncoder:
    def __init__(
        self, 
        seed: int,
        model_dim: int,
        hidden_dim: int,
        num_attention_layers: int,
        dk: int,
        dv: int
    ):
        self.attention = Attention(seed, model_dim, num_attention_layers, dk, dv)
        self.network = FeedForwardNetwork(seed, model_dim, hidden_dim)
        
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
        seed: int,
        input_dim: int,
        hidden_dim: int,
        num_attention_layers: int,
        dk: int,
        dv: int,
        num_encoder: int = 6
    ):
        self.layers = [SingleEncoder(seed, input_dim, hidden_dim, num_attention_layers, dk, dv) for _ in range(num_encoder)]

    def encode(self, x: jnp.ndarray, mask: jnp.ndarray = None) -> jnp.ndarray:
        y = x
        for layer in self.layers:
            y = layer.encode(y, mask)
        return y

# Single decoder will have two multi-head attentions and one feed forward NN
class SingleDecoder:
    def __init__(
        self,
        seed: int,
        model_dim: int,
        hidden_dim: int,
        num_att_layers: int,
        dk: int,
        dv: int
    ):
        self.first_attention = Attention(seed, model_dim, num_att_layers, dk, dv)
        self.second_attention = Attention(seed, model_dim, num_att_layers, dk, dv)
        self.network = FeedForwardNetwork(seed, model_dim, hidden_dim)

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
        y = self.network.forward(att_val2)
        # Add + normalize
        y = basic_normalize(att_val2 + y)

        return y

class Decoder:
    def __init__(
        self,
        seed: int,
        model_dim: int,
        hidden_dim: int,
        num_att_layers: int,
        dk: int,
        dv: int,
        num_decoder: int = 6
    ):
        self.layers = [SingleDecoder(seed, model_dim, hidden_dim, num_att_layers, dk, dv) for _ in range(num_decoder)]

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
        seed: int,
        num_vocab: int,
        model_dim: int,
        hidden_dim: int,
        num_attention_layers: int,
        dk: int,
        dv: int,
        max_seq_len: int = 5000,
        num_encoder: int = 6,
        num_decoder: int = 6
    ):
        self.encode_preprocessor = Preprocessing(seed, num_vocab, model_dim, max_seq_len)
        self.encoder = Encoder(seed, model_dim, hidden_dim, num_attention_layers, dk, dv, num_encoder)

        self.decode_preprocessor = Preprocessing(seed, num_vocab, model_dim, max_seq_len)
        self.decoder = Decoder(seed, model_dim, hidden_dim, num_attention_layers, dk, dv, num_decoder)

    def forward(self,
        input: jnp.ndarray,
        output: jnp.ndarray,
        input_mask: jnp.ndarray = None,
        output_mask: jnp.ndarray = None
    ) -> jnp.ndarray:
        preprocessed_input = self.encode_preprocessor.preprocess(input)
        encoding_mem = self.encoder.encode(preprocessed_input, input_mask)

        preprocessed_output = self.decode_preprocessor.preprocess(output)
        return self.decoder.decode(preprocessed_output, encoding_mem, input_mask, output_mask)