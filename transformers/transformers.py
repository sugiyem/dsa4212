import numpy as np
import torch 
import torch.nn as nn 
from transformer_train import subsequent_mask

class Attention(nn.Module):
    """
    Attention class for the transformer implementation.
    """
    def __init__(self, 
        model_dim: int, 
        num_attention_layer: int,
        dk: int,
        dv: int
    ):
        super().__init__()

        self.model_dim = model_dim 
        self.num_attention_layer = num_attention_layer 
        self.dk = dk 
        self.dv = dv

        # Bias for wq, wk, wv weight must be 0, as it should represent a matrix multiplication
        self.wq = nn.Linear(model_dim, num_attention_layer * dk, bias=False)
        self.wk = nn.Linear(model_dim, num_attention_layer * dk, bias=False)
        self.wv = nn.Linear(model_dim, num_attention_layer * dv, bias=False)

        self.wo = nn.Linear(self.num_attention_layer * dv, self.model_dim)

    @staticmethod
    def calc_attention(
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculates the attention from query, key, and value.
        
        Args:
            query (torch.Tensor): A Tensor of size (num_data, seq_len, model_dim).
            key (torch.Tensor) : A Tensor of size (num_data, seq_len, model_dim).
            value (torch.Tensor): A Tensor of size (num_data, seq_len, model_dim).
        
        Returns:
            The computed attention from query, key, and value.
        """
        dk = query.shape[-1]
        scores = torch.matmul(query, key.transpose(-1,-2)) / (dk**0.5)
        return torch.matmul(nn.functional.softmax(scores, dim=-1), value)

    @staticmethod 
    def calc_masked_attention(
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor, 
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculates the masked attention from query, key, and value.
        
        Args:
            query (torch.Tensor): A Tensor of size (num_data, seq_len, model_dim).
            key (torch.Tensor) : A Tensor of size (num_data, seq_len, model_dim).
            value (torch.Tensor): A Tensor of size (num_data, seq_len, model_dim).
            mask (torch.Tensor): A Tensor representing the mask, of size (num_data, seq_len, seq_len).
        
        Returns:
            The computed masked attention from query, key, and value.
        """
        dk = query.shape[-1]
        scores = torch.matmul(query, key.transpose(-1,-2)) / (dk**0.5)
        masked_scores = torch.where(mask == 0, -1e9, scores)
        return torch.matmul(nn.functional.softmax(masked_scores, dim=-1), value)
    
    def forward(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor, 
        mask : torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward method of the Attention class.

        Args:
            query (torch.Tensor): A Tensor of size (num_data, len_seq, dim_size).
            key (torch.Tensor): A Tensor of size (num_data, len_seq, dim_size).
            value (torch.Tensor): A Tensor of size (num_data, len_seq, dim_size).
            mask (torch.Tensor): A Tensor of size (num_data, d1, d2) if not None.
        """
        
        num_data = query.shape[0]

        # convert mask to 4D tensor
        if mask is not None:
            mask = mask.reshape(mask.shape[0], 1, mask.shape[1], mask.shape[2])

        # All these transformed fields will be a tensor of size (num_data, seq_len, num_attention_layer, attention_dim)
        transformed_query = self.wq(query).reshape(num_data, -1, self.num_attention_layer, self.dk).transpose(1, 2)
        transformed_key = self.wk(key).reshape(num_data, -1, self.num_attention_layer, self.dk).transpose(1, 2)
        transformed_value = self.wv(value).reshape(num_data, -1, self.num_attention_layer, self.dv).transpose(1, 2)

        if mask is None:
            attention = self.calc_attention(transformed_query, transformed_key, transformed_value)
        else:
            attention = self.calc_masked_attention(transformed_query, transformed_key, transformed_value, mask)
        
        # Transform the shape of attention to (num_data, len_seq, dim_size)
        attention = attention.transpose(1,2).reshape(num_data, -1, self.model_dim)
        
        return self.wo(attention)

# MLP with 1 hidden layer and ReLU as it's activation function
class FeedForwardNetwork(nn.Module):
    """
    Multi-layer perceptron with 1 hidden layer and ReLU as its activation function.
    """
    def __init__(self, model_dim: int, feedforward_dim: int):
        super().__init__()
        self.dense1 = nn.Linear(model_dim, feedforward_dim)
        self.dense2 = nn.Linear(feedforward_dim, model_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward method of the FeedForwardNetwork class.

        Args:
            x (torch.Tensor): Input Tensor of size (num_data, model_dim).

        Returns:
            An output Tensor of size (num_data, model_dim).
        """
        x = self.dense1(x)
        x = nn.functional.relu(x)
        x = self.dense2(x)
        return x

class Embedding(nn.Module):
    """
    Class to represent an embedding.
    """
    def __init__(self, num_vocab: int, model_dim: int):
        super().__init__()
        self.model_dim = model_dim
        self.embed = nn.Embedding(num_vocab, model_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward method of the Embedding class.

        Args:
            x (torch.Tensor): Input Tensor of size (num_data, seq_len).
        
        Returns:
            A Tensor that has been passed through the `embed` layer and scaled,
            with a size of (num_data, seq_len, model_dim).
        """
        return self.embed(x) * (self.model_dim ** 0.5)

class PositionalEncoder(nn.Module):
    """
    Class representing a positional encoder.
    """
    def __init__(self, model_dim: int, max_seq_len: int):
        super().__init__()
        assert model_dim % 2 == 0, "model dimension must be even"

        position = np.arange(max_seq_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, model_dim, 2) * -(np.log(10000.0) / model_dim))
        
        self.position_encoding = np.zeros((max_seq_len, model_dim))
        self.position_encoding[:, 0::2] = np.sin(position * div_term)
        self.position_encoding[:, 1::2] = np.cos(position * div_term)
        self.position_encoding = torch.from_numpy(self.position_encoding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        """
        # input x is a tensor of size (num_data, seq_len, model_dim)
        # will output a tensor of size (num_data, seq_len, model_dim)
        seq_len = x.shape[1]
        position_encoding = self.position_encoding[:seq_len, :].to(x.dtype)
        return x + position_encoding
        
# Represents the whole pre-processing step before the real encoding steps in
# Embeddding + Positional Encoding
class Preprocessing(nn.Module):
    def __init__(self, num_vocab: int, model_dim: int, max_seq_len: int):
        super().__init__()
        self.embedding = Embedding(num_vocab, model_dim)
        self.positional_encoder = PositionalEncoder(model_dim, max_seq_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = self.positional_encoder(x)
        return x
    
# Single encoder will have one multi-head attention and one feed forward NN
class SingleEncoder(nn.Module):
    def __init__(self, 
        model_dim: int, 
        feedforward_dim: int, 
        num_attention_layer: int,
        attention_dk: int,
        attention_dv: int
    ):
        super().__init__()
        self.attention = Attention(model_dim, num_attention_layer, attention_dk, attention_dv)
        self.network = FeedForwardNetwork(model_dim, feedforward_dim)
        self.normalizer1 = nn.LayerNorm(model_dim)
        self.normalizer2 = nn.LayerNorm(model_dim)
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        normed_x = self.normalizer1(x)
        att_val = self.attention(normed_x, normed_x, normed_x, mask)
        x = x + att_val 

        normed_x = self.normalizer2(x)
        net_val = self.network(normed_x)
        x = x + net_val

        return x
    
class Encoder(nn.Module):
    def __init__(self, 
        model_dim: int, 
        feedforward_dim: int,
        num_attention_layer: int, 
        attention_dk: int,
        attention_dv: int,
        num_encoder: int
    ):
        super().__init__()
        self.layers = nn.ModuleList([SingleEncoder(model_dim, feedforward_dim, num_attention_layer, \
                       attention_dk, attention_dv) for _ in range(num_encoder)])
        self.normalizer = nn.LayerNorm(model_dim)
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        x = self.normalizer(x)
        return x

# Single decoder will have two multi-head attentions and one feed forward NN
class SingleDecoder(nn.Module):
    def __init__(self, 
        model_dim: int, 
        feedforward_dim: int, 
        num_attention_layer: int,
        attention_dk: int,
        attention_dv: int
    ):
        super().__init__()
        self.attention1 = Attention(model_dim, num_attention_layer, attention_dk, attention_dv)
        self.attention2 = Attention(model_dim, num_attention_layer, attention_dk, attention_dv)
        self.network = FeedForwardNetwork(model_dim, feedforward_dim)
        self.normalizer1 = nn.LayerNorm(model_dim)
        self.normalizer2 = nn.LayerNorm(model_dim)
        self.normalizer3 = nn.LayerNorm(model_dim)

    def forward(self, 
        x: torch.Tensor, 
        encoding_mem: torch.Tensor, 
        src_mask: torch.Tensor = None, 
        tgt_mask: torch.Tensor = None
    ) -> torch.Tensor:
        normed_x = self.normalizer1(x)
        att_val1 = self.attention1(normed_x, normed_x, normed_x, tgt_mask)
        x = x + att_val1 

        normed_x = self.normalizer2(x)
        att_val2 = self.attention2(normed_x, encoding_mem, encoding_mem, src_mask)
        x = x + att_val2 

        normed_x = self.normalizer3(x)
        net_val = self.network(normed_x)
        x = x + net_val 
        
        return x
        

class Decoder(nn.Module):
    def __init__(self, 
        model_dim: int, 
        feedforward_dim: int, 
        num_attention_layer: int, 
        attention_dk: int,
        attention_dv: int,
        num_decoder: int
    ):
        super().__init__()
        self.layers = nn.ModuleList([SingleDecoder(model_dim, feedforward_dim, num_attention_layer, \
                       attention_dk, attention_dv) for _ in range(num_decoder)])
        self.normalizer = nn.LayerNorm(model_dim)
        
    def forward(self, 
        x: torch.Tensor, 
        encoding_mem: torch.Tensor, 
        src_mask: torch.Tensor = None, 
        tgt_mask: torch.Tensor = None
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, encoding_mem, src_mask, tgt_mask)
        x = self.normalizer(x)
        return x
    
class LogitsGenerator(nn.Module):
    def __init__(self, model_dim: int, num_vocab: int):
        super().__init__()
        self.dense = nn.Linear(model_dim, num_vocab)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dense(x)
        return nn.functional.log_softmax(x, dim=-1)
    
class Transformer(nn.Module):
    def __init__(self, 
        input_vocab: int, 
        output_vocab: int, 
        model_dim: int = 512, 
        feedforward_dim: int = 2048, 
        num_attention_layer: int = 8, 
        attention_dk: int = 64,
        attention_dv: int = 64,
        max_seq_len: int = 5000, 
        num_coder: int = 6
    ):
        super().__init__()

        self.encode_preprocessor = Preprocessing(input_vocab, model_dim, max_seq_len)
        self.encoder = Encoder(model_dim, feedforward_dim, num_attention_layer, attention_dk, attention_dv, num_coder)
        
        self.decode_preprocessor = Preprocessing(output_vocab, model_dim, max_seq_len)
        self.decoder = Decoder(model_dim, feedforward_dim, num_attention_layer, attention_dk, attention_dv, num_coder)
        
        self.generator = LogitsGenerator(model_dim, output_vocab)

        # Initialize parameters with xavier uniform
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
    
    def forward(self, 
        input: torch.Tensor, 
        output: torch.Tensor, 
        input_mask: torch.Tensor = None, 
        output_mask: torch.Tensor = None
    ) -> torch.Tensor:
        preprocessed_input = self.encode_preprocessor(input)
        encoding_mem = self.encoder(preprocessed_input, input_mask)

        preprocessed_output = self.decode_preprocessor(output)
        out = self.decoder(preprocessed_output, encoding_mem, input_mask, output_mask)

        logits = self.generator(out)
        
        return logits
    
    def greedy_decode(
        self,
        input: torch.Tensor, 
        output_init: torch.Tensor,
        output_len: int,
        input_mask: torch.Tensor = None
    ) -> torch.Tensor:
        self.eval()
        curr_output = output_init 

        for _ in range(output_len - 1):
            logits = self.forward(input, curr_output, input_mask, subsequent_mask(curr_output.shape[1]))

            # only consider output for last sequence 
            prob = logits[:, -1, :]
            # greedy decoding: pick the vocab with largest probability 
            _, next_vocab = torch.max(prob, dim=1)

            curr_output = torch.hstack((curr_output, next_vocab.reshape(-1, 1)))

        return curr_output
