import numpy as np
import torch 
import torch.nn as nn 

# masking function given in https://nlp.seas.harvard.edu/2018/04/03/attention.html
def subsequent_mask(size: int) -> torch.Tensor:
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

# batch data given in https://nlp.seas.harvard.edu/2018/04/03/attention.html
class Batch:
    "Object for holding a batch of data with mask during training."
    def __init__(self, 
        src: torch.Tensor, 
        trg: torch.Tensor, 
        pad: int = 0
    ):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        self.trg = trg[:, :-1]
        self.trg_y = trg[:, 1:]
        self.trg_mask = self.make_std_mask(self.trg, pad)
        self.ntokens = (self.trg_y != pad).data.sum()
    
    @staticmethod
    def make_std_mask(tgt: torch.Tensor, pad: int) -> torch.Tensor:
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1))
        return tgt_mask
    
# The learning rate specified in the paper https://arxiv.org/abs/1706.03762
# The idea for this implementation is from https://nlp.seas.harvard.edu/2018/04/03/attention.html
class NoamOptimizer:
    "Optim wrapper that implements rate."
    def __init__(self, 
        optimizer: torch.optim.Optimizer,
        model_dim: int,
        factor: float, 
        warmup: float, 
    ):
        self.optimizer = optimizer
        self.model_dim = model_dim 
        self.factor = factor
        self.warmup = warmup
        self.curr_step = 0
        
    def step(self):
        "Update parameters and rate"
        self.curr_step += 1
        rate = self.factor * (self.model_dim ** -0.5) * min(self.curr_step ** -0.5, self.curr_step * self.warmup ** -1.5)

        for param in self.optimizer.param_groups:
            param['lr'] = rate

        self.optimizer.step()