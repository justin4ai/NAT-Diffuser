import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_
import warnings
import sys
from rotary_embedding_torch import RotaryEmbedding
from natten.functional import na2d_av, na2d_qk, na1d_av, na1d_qk, na1d
from natten import (
    use_fused_na,
    use_autotuner
)
use_fused_na(True)
use_autotuner(True)


class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, H):
        super().__init__()
        assert H.bert_n_emb % H.bert_n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(H.bert_n_emb, H.bert_n_emb)
        self.query = nn.Linear(H.bert_n_emb, H.bert_n_emb)
        self.value = nn.Linear(H.bert_n_emb, H.bert_n_emb)
        # regularization
        self.attn_drop = nn.Dropout(H.attn_pdrop)
        self.resid_drop = nn.Dropout(H.resid_pdrop)
        # output projection
        self.proj = nn.Linear(H.bert_n_emb, H.bert_n_emb)
        self.n_head = H.bert_n_head
        self.causal = True if H.sampler == 'autoregressive' else False
        if self.causal:
            block_size = np.prod(H.latent_shape)
            mask = torch.tril(torch.ones(block_size, block_size))
            self.register_buffer("mask", mask.view(1, 1, block_size, block_size))

    def forward(self, x, layer_past=None):

        B, T, C = x.size() #torch.Size([20, 256, 512])

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs) # torch.Size([20, 8, 256, 64])
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        present = torch.stack((k, v)) # torch.Size([2, 20, 8, 256, 64])
        if self.causal and layer_past is not None:
            past_key, past_value = layer_past
            k = torch.cat((past_key, k), dim=-2)
            v = torch.cat((past_value, v), dim=-2)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) # torch.Size([20, 8, 256, 256])

        if self.causal and layer_past is None:
            print("inf!!")
            att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))

        att = F.softmax(att, dim=-1) # torch.Size([20, 8, 256, 256])
        att = self.attn_drop(att) # torch.Size([20, 8, 256, 256])
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        # re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # output projection
        y = self.resid_drop(self.proj(y)) # torch.Size([20, 256, 512])
        return y, present
    
class Hydra1DNeighborhoodAttention(nn.Module):
    def __init__(self, H):
        super().__init__()
        assert H.bert_n_emb % H.bert_n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(H.bert_n_emb, H.bert_n_emb)
        self.query = nn.Linear(H.bert_n_emb, H.bert_n_emb)
        self.value = nn.Linear(H.bert_n_emb, H.bert_n_emb)
        # regularization
        self.attn_drop = nn.Dropout(H.attn_pdrop)
        self.resid_drop = nn.Dropout(H.resid_pdrop)
        # output projection
        self.proj = nn.Linear(H.bert_n_emb, H.bert_n_emb)
        self.n_head = H.bert_n_head
        self.causal = True if H.sampler == 'autoregressive' else False

        self.dilations = [1, 4, 8, 16] # Ensure each element must be a divisor of self.n_heads
        self.kernel_sizes = [7] * len(self.dilations)
        self.num_splits = len(self.kernel_sizes)


        self.use_rpb = False

        if self.causal:
            block_size = np.prod(H.latent_shape)
            mask = torch.tril(torch.ones(block_size, block_size))
            self.register_buffer("mask", mask.view(1, 1, block_size, block_size))


        if self.n_head % len(self.kernel_sizes) == 0:
            self.clean_partition = True
        else:
            print("Error: Adaptive handling of discrepancy between n_heads and the number of kernel sizes has not been implemented yet.")
            sys.exit(1) 
            
        
        self.rpb = nn.ParameterList([nn.Parameter(torch.zeros(self.n_head//self.num_splits + 1, (2*k-1), (2*k-1))) for k in self.kernel_sizes])
        self.rotary_emb = RotaryEmbedding(dim = 32)

    def forward(self, x, layer_past=None):

        B, T, C = x.size() #torch.Size([20, 256, 512])

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs) # torch.Size([20, 8, 256, 64])
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        B, nh, T, hs = q.size()

        #########################
        ## - sync 1 finishes - ##
        #########################

        present = torch.stack((k, v)) # torch.Size([2, 20, 8, 256, 64]) / not used in our code
        if self.causal and layer_past is not None:
            print("?")
            past_key, past_value = layer_past
            k = torch.cat((past_key, k), dim=-2)
            v = torch.cat((past_value, v), dim=-2)


        if self.clean_partition: # Always true for this implementation

            #print(f"num_splits : {self.num_splits}") # with low-resolution, 1 / high-resolutions : 2
            q = q.chunk(self.num_splits, dim=1) # returns self.num_splits of tuple splitted in num_heads -> different multihead groups!
            k = k.chunk(self.num_splits, dim=1)
            v = v.chunk(self.num_splits, dim=1)

        if self.use_rpb:
            attention = [na1d_qk(_q, _k,
                        kernel_size=_kernel_size,
                        dilation=_dilation,
                        rpb=_rpb) \
                for _q,_k,_rpb,_kernel_size,_dilation in zip(q, k, self.rpb, self.kernel_sizes, self.dilations)]
        else:

            q = self.rotary_emb.rotate_queries_or_keys(q)
            k = self.rotary_emb.rotate_queries_or_keys(k)
            v = self.rotary_emb.rotate_queries_or_keys(v)

            attention = [na1d(_q, _k, _v, kernel_size = _kernel_size, dilation = _dilation) \
                         for _q, _k, _v, _kernel_size, _dilation in zip(q, k, v, self.kernel_sizes, self.dilations) ]



        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        #att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) # torch.Size([20, 8, 256, 256])
        #B, nh, T, hs = q.size()


        # print(q[0].size())
        # print(k[0].size())
        # print(self.rpb[0].size())
        # print(self.kernel_sizes[0])
        # print(self.dilations[0])

        #attention = [na1d_qk(_q.permute(0, 2 ,1, 3), _k.permute(0, 2 ,1, 3),


        attention = [a.softmax(dim=-1) for a in attention]
        attention = [self.attn_drop(a) for a in attention]

        x = [na2d_av(_attn, _v,
                     kernel_size=_k,
                     dilation=_d) \
             for _attn, _v, _k, _d in zip(attention, v, self.kernel_sizes, self.dilations)]

        y = torch.cat(x, dim=1)

        if self.causal and layer_past is None: # not used in this implementation
            print("inf!!")
            att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))



        # att = F.softmax(att, dim=-1) # torch.Size([20, 8, 256, 256])
        # att = self.attn_drop(att) # torch.Size([20, 8, 256, 256])
        #y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        # re-assemble all head outputs side by side
        print(f"y shape : {y.size()}")
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # output projection
        y = self.resid_drop(self.proj(y)) # torch.Size([20, 256, 512])
        return y, present

class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, H):
        super().__init__()
        self.ln1 = nn.LayerNorm(H.bert_n_emb)
        self.ln2 = nn.LayerNorm(H.bert_n_emb)
        #self.attn = CausalSelfAttention(H)
        
        h_params = H.items()
        # Save parameters of H
        with open("param.txt", "w") as file:
            for key, value in h_params:
                file.write(f"{key}: {value}\n")

        self.attn = Hydra1DNeighborhoodAttention(H)
        self.mlp = nn.Sequential(
            nn.Linear(H.bert_n_emb, 4 * H.bert_n_emb),
            nn.GELU(),  # nice
            nn.Linear(4 * H.bert_n_emb, H.bert_n_emb),
            nn.Dropout(H.resid_pdrop),
        )

    def forward(self, x, layer_past=None, return_present=False):

        attn, present = self.attn(self.ln1(x), layer_past) # torch.Size([20, 256, 512])

        x = x + attn # torch.Size([20, 256, 512])

        x = x + self.mlp(self.ln2(x)) # torch.Size([20, 256, 512])


        if layer_past is not None or return_present:
            return x, present
        return x # torch.Size([20, 256, 512])


class Transformer(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, H):
        super().__init__()

        self.vocab_size = H.codebook_size + 1 # 1025
        self.n_embd = H.bert_n_emb
        self.block_size = H.block_size
        self.n_layers = H.bert_n_layers
        self.codebook_size = H.codebook_size
        self.causal = H.sampler == 'autoregressive'
        if self.causal:
            self.vocab_size = H.codebook_size

        self.tok_emb = nn.Embedding(self.vocab_size, self.n_embd) # 512-dimensional emb
        self.pos_emb = nn.Parameter(
            torch.zeros(1, self.block_size, self.n_embd))
        self.start_tok = nn.Parameter(torch.zeros(1, 1, self.n_embd))
        self.drop = nn.Dropout(H.embd_pdrop)

        # transformer
        self.blocks = nn.Sequential(*[Block(H) for _ in range(self.n_layers)])
   
        # decoder head
        self.ln_f = nn.LayerNorm(self.n_embd)
        self.head = nn.Linear(self.n_embd, self.codebook_size, bias=False)

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, idx, t=None): # torch.Size([b, 256]) - 
        # each index maps to a (learnable) vector

        token_embeddings = self.tok_emb(idx) # torch.Size([20, 256, 512])

        if self.causal:
            token_embeddings = torch.cat(
                (self.start_tok.repeat(token_embeddings.size(0), 1, 1), token_embeddings),
                dim=1
            )

        t = token_embeddings.shape[1]
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."
        # each position maps to a (learnable) vector

        position_embeddings = self.pos_emb[:, :t, :]

        x = token_embeddings + position_embeddings
        x = self.drop(x)

        for block in self.blocks:
            # torch.Size([20, 256, 512])
            x = block(x)
            # torch.Size([20, 256, 512])

        x = self.ln_f(x)
        logits = self.head(x)

        return logits # torch.Size([20, 256, 1024])
