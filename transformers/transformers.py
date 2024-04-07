import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn #Linen API
from utils import InputParams, MultiHeadAttnParams, softmax, rng_unif, relu, basic_normalize

class Attention(nn.Module):
    model_dim: int 
    num_attention_layer: int 

    def setup(self):
        assert self.model_dim % self.num_attention_layer == 0, "model dimension must be divisible by number of layers"

        self.attention_dim = self.model_dim // self.num_attention_layer # key, query and value dimension
        self.wq = nn.Dense(self.model_dim)
        self.wk = nn.Dense(self.model_dim)
        self.wv = nn.Dense(self.model_dim)
        self.wo = nn.Dense(self.model_dim)

    @staticmethod
    @jax.jit
    def calc_attention(
        query: jnp.ndarray,
        key: jnp.ndarray,
        value: jnp.ndarray,
    ) -> jnp.ndarray:
        dk = query.shape[-1]
        scores = jnp.matmul(query, key.transpose(0,1,3,2)) / jnp.sqrt(dk)
        return nn.softmax(scores) @ value

    @staticmethod 
    @jax.jit
    def calc_masked_attention(
        query: jnp.ndarray,
        key: jnp.ndarray,
        value: jnp.ndarray,
        mask: jnp.ndarray
    ) -> jnp.ndarray:
        dk = query.shape[-1]
        scores = jnp.matmul(query, key.transpose(0,1,3,2)) / jnp.sqrt(dk)
        masked_scores = jnp.where(mask == 0, -1e9, scores)
        return nn.softmax(masked_scores) @ value
    
    def __call__(
        self,
        query: jnp.ndarray,
        key: jnp.ndarray,
        value: jnp.ndarray,
        mask: jnp.ndarray = None    
    ) -> jnp.ndarray:
        # query, key, value must all be a jnp.array of size (num_data, len_seq, dim_size)
        # if mask is not None, it is a 3D jnp.array of size (num_data, d1, d2) for some d1, d2
        # it's okay for len_seq to be different during decoding process
        num_data = query.shape[0]

        # convert mask to 4D array
        if mask is not None:
            mask = mask.reshape(mask.shape[0], 1, mask.shape[1], mask.shape[2])

        # All these transformed fields will be a jnp.ndarray of size (num_data, num_att_ayer, len_seq, att_dim)
        transformed_query = self.wq(query).reshape(num_data, -1, self.num_attention_layer, self.attention_dim).transpose(0,2,1,3)
        transformed_key = self.wk(key).reshape(num_data, -1, self.num_attention_layer, self.attention_dim).transpose(0,2,1,3)
        transformed_value = self.wv(value).reshape(num_data, -1, self.num_attention_layer, self.attention_dim).transpose(0,2,1,3)

        attention = self.calc_attention(transformed_query, transformed_key, transformed_value) if mask is None \
            else self.calc_masked_attention(transformed_query, transformed_key, transformed_value, mask)
        
        # Transform the shape of attention to (num_data, len_seq, dim_size)
        attention = attention.transpose(0,2,1,3).reshape(num_data, -1, self.model_dim)
        return self.wo(attention) # the output is a jnp.array of size (num_data, len_seq, dim_size)

# MLP with 1 hidden layer and ReLU as it's activation function
class FeedForwardNetwork(nn.Module):
    model_dim: int 
    feedforward_dim: int

    def setup(self):
        self.dense1 = nn.Dense(features=self.feedforward_dim)
        self.dense2 = nn.Dense(features=self.model_dim)

    def __call__(self, x):
        x = self.dense1(x)
        x = nn.relu(x)
        x = self.dense2(x)
        return x

class Embedding(nn.Module):
    num_vocab: int 
    model_dim: int

    def setup(self):
        self.embed = nn.Embed(num_embeddings=self.num_vocab, features=self.model_dim)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # input x is a jnp.ndarray of size (num_data, seq_len)
        # will output a jnp.ndarray of size (num_data, seq_len, model_dim)
        return self.embed(x) * jnp.sqrt(self.model_dim)

class PositionalEncoder(nn.Module):
    model_dim: int 
    max_seq_len: int = 5000

    def setup(self):
        assert self.model_dim % 2 == 0, "model dimension must be even"

        position = np.arange(self.max_seq_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, self.model_dim, 2) * -(np.log(10000.0) / self.model_dim))
        
        self.position_encoding = np.zeros((self.max_seq_len, self.model_dim))
        self.position_encoding[:, 0::2] = np.sin(position * div_term)
        self.position_encoding[:, 1::2] = np.cos(position * div_term)
        self.position_encoding = jnp.array(self.position_encoding)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # input x is a jnp.ndarray of size (num_data, seq_len, model_dim)
        # will output a jnp.ndarray of size (num_data, seq_len, model_dim)
        seq_len = x.shape[1]
        return x + self.position_encoding[:seq_len, :] 
        

# Represents the whole pre-processing step before the real encoding steps in
# Embeddding + Positional Encoding
class Preprocessing(nn.Module):
    num_vocab: int 
    model_dim: int 
    max_seq_len: int = 5000

    def setup(self):
        self.embedding = Embedding(self.num_vocab, self.model_dim)
        self.positional_encoder = PositionalEncoder(self.model_dim, self.max_seq_len)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = self.embedding(x)
        x = self.positional_encoder(x)
        return x
    
# Single encoder will have one multi-head attention and one feed forward NN
class SingleEncoder(nn.Module):
    model_dim: int 
    feedforward_dim: int
    num_attention_layer: int 

    def setup(self):
        self.attention = Attention(self.model_dim, self.num_attention_layer)
        self.network = FeedForwardNetwork(self.model_dim, self.feedforward_dim)
        self.normalizer1 = nn.LayerNorm()
        self.normalizer2 = nn.LayerNorm()
        
    def __call__(self, x: jnp.ndarray, mask: jnp.ndarray = None) -> jnp.ndarray:
        # Use the multi head attention
        att_val = self.attention(x, x, x, mask) 
        # Normalize
        x = self.normalizer1(x + att_val)

        # Use the feed forward NN
        net_val = self.network(x)
        # Normalize again
        x = self.normalizer2(x + net_val)

        return x
    
class Encoder(nn.Module):
    model_dim: int 
    feedforward_dim: int 
    num_attention_layer: int 
    num_encoder: int = 6

    def setup(self):
        self.layers = [SingleEncoder(self.model_dim, self.feedforward_dim, self.num_attention_layer) \
                       for _ in range(self.num_encoder)]
        
    def __call__(self, x: jnp.ndarray, mask: jnp.ndarray = None) -> jnp.ndarray:
        for layer in self.layers:
            x = layer(x, mask)
        return x


# Single decoder will have two multi-head attentions and one feed forward NN
class SingleDecoder(nn.Module):
    model_dim: int
    feedforward_dim: int 
    num_attention_layer: int 

    def setup(self):
        self.attention1 = Attention(self.model_dim, self.num_attention_layer)
        self.attention2 = Attention(self.model_dim, self.num_attention_layer)
        self.network = FeedForwardNetwork(self.model_dim, self.feedforward_dim)
        self.normalizer1 = nn.LayerNorm()
        self.normalizer2 = nn.LayerNorm()
        self.normalizer3 = nn.LayerNorm()

    def __call__(self,
        x: jnp.ndarray,
        encoding_mem: jnp.ndarray, # memory during encoding process
        mask1: jnp.ndarray = None,
        mask2: jnp.ndarray = None
    ) -> jnp.ndarray:
        # Use the first multi-head attention
        att_val1 = self.attention1(x, x, x, mask1)
        # Normalize
        x = self.normalizer1(x + att_val1)

        # Use the second multi-head attention
        att_val2 = self.attention2(x, encoding_mem, encoding_mem, mask2)
        # Normalize
        x = self.normalizer2(x + att_val2)

        # Use the feed forward NN
        net_val = self.network(x)
        # Normalize
        x = self.normalizer3(x + net_val)

        return x
        

class Decoder(nn.Module):
    model_dim: int 
    feedforward_dim: int 
    num_attention_layer: int 
    num_decoder: int = 6

    def setup(self):
        self.layers = [SingleDecoder(self.model_dim, self.feedforward_dim, self.num_attention_layer) \
                       for _ in range(self.num_decoder)]
    def __call__(self,
        x: jnp.ndarray,
        encoding_mem: jnp.ndarray,
        mask1: jnp.ndarray = None,
        mask2: jnp.ndarray = None
    ) -> jnp.ndarray:
        for layer in self.layers:
            x = layer(x, encoding_mem, mask1, mask2)
        return x
    
class LogitsGenerator(nn.Module):
    num_vocab: int 

    def setup(self):
        self.dense = nn.Dense(features=self.num_vocab)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = self.dense(x)
        return nn.log_softmax(x)
    
class Transformer(nn.Module):
    input_vocab: int
    output_vocab: int 
    model_dim: int 
    feedforward_dim: int 
    num_attention_layer: int 
    max_seq_len: int = 5000
    num_coder: int = 6

    def setup(self):
        self.encode_preprocessor = Preprocessing(self.input_vocab, self.model_dim, self.max_seq_len)
        self.encoder = Encoder(self.model_dim, self.feedforward_dim, self.num_attention_layer, self.num_coder)
        
        self.decode_preprocessor = Preprocessing(self.output_vocab, self.model_dim, self.max_seq_len)
        self.decoder = Decoder(self.model_dim, self.feedforward_dim, self.num_attention_layer, self.num_coder)
        
        self.generator = LogitsGenerator(self.output_vocab)
    
    def __call__(self,
        input: jnp.ndarray,
        output: jnp.ndarray,
        input_mask: jnp.ndarray = None,
        output_mask: jnp.ndarray = None
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        # input is a jnp.ndarray of size (num_data, input_dim)
        # output is a jnp.ndarray of size (num_data, output_dim)
        preprocessed_input = self.encode_preprocessor(input)
        encoding_mem = self.encoder(preprocessed_input, input_mask)
        preprocessed_output = self.decode_preprocessor(output)
        out = self.decoder(preprocessed_output, encoding_mem, output_mask, input_mask)
        logits = self.generator(out)
        
        return logits