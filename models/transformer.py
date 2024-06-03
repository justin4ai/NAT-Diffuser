import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_
import warnings
from natten.functional import natten2dqkrpb, natten2dav
# from natten import (
#       #enable_fused_na,
#       disable_fused_na,
#       #enable_autotuner,
#       disable_autotuner
# )
# enable_fused_na()
# enable_autotuner()


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
    
class HydraNeighborhoodAttention(nn.Module):
    def __init__(self,
                 dim,
                 kernel_sizes, # Array for kernel sizes
                 num_heads,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 dilations=[1], # Array of dilations
                 ):
        super().__init__()
        if len(kernel_sizes) == 1 and len(dilations) != 1:
            kernel_sizes = [kernel_sizes[0] for _ in range(len(dilations))]
        elif len(dilations) == 1 and len(kernel_sizes) != 1:
            dilations = [dilations[0] for _ in range(len(kernel_sizes))]
        assert(len(kernel_sizes) == len(dilations)),f"Number of kernels ({(kernel_sizes)}) must be the same as number of dilations ({(dilations)})"
        self.num_splits = len(kernel_sizes)
        self.num_heads = num_heads
        self.kernel_sizes = kernel_sizes
        self.dilations = dilations

        self.head_dim = dim // self.num_heads
        self.scale = qk_scale or self.head_dim ** -0.5

        asserts = []
        for i in range(len(kernel_sizes)):
            asserts.append(kernel_sizes[i] > 1 and kernel_sizes[i] % 2 == 1)
            if asserts[i] == False:
                warnings.warn(f"Kernel_size {kernel_sizes[i]} needs to be >1 and odd")
        assert(all(asserts)),f"Kernel sizes must be >1 AND odd. Got {kernel_sizes}"

        self.window_size = []
        for i in range(len(dilations)):
            self.window_size.append(self.kernel_sizes[i] * self.dilations[i])

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        # Needs to be fixed if we want uneven head splits. // is floored
        # division
        if num_heads % len(kernel_sizes) == 0:
            self.rpb = nn.ParameterList([nn.Parameter(torch.zeros(num_heads//self.num_splits, (2*k-1), (2*k-1))) for k in kernel_sizes])
            self.clean_partition = True
        else:
            diff = num_heads - self.num_splits * (num_heads // self.num_splits)
            rpb = [nn.Parameter(torch.zeros(num_heads//self.num_splits, (2*k-1), (2*k-1))) for k in kernel_sizes[:-diff]]
            for k in kernel_sizes[-diff:]:
                rpb.append(nn.Parameter(torch.zeros(num_heads//self.num_splits + 1, (2*k-1), (2*k-1))))
            assert(sum(r.shape[0] for r in rpb) == num_heads),f"Got {sum(r.shape[0] for r in rpb)} heads."
            self.rpb = nn.ParameterList(rpb)

            self.clean_partition = False
            self.shapes = [r.shape[0] for r in rpb]
            warnings.warn(f"Number of partitions ({self.num_splits}) do not "\
                    f"evenly divide the number of heads ({self.num_heads}). "\
                    f"We evenly divide the remainder between the last {diff} "\
                    f"heads This may cause unexpected behavior. Your head " \
                    f"partitions look like {self.shapes}")

        [trunc_normal_(rpb, std=0.02, mean=0.0, a=-2., b=2.) for rpb in self.rpb]
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, Hp, Wp, C = x.shape
        H, W = Hp, Wp
        qkv = self.qkv(x).reshape(B, H, W, 3, self.num_heads, self.head_dim).permute(3, 0, 4, 1, 2, 5)

        q, k, v = qkv.chunk(3, dim=0)
        q = q.squeeze(0) * self.scale
        k = k.squeeze(0)
        v = v.squeeze(0)

        if self.clean_partition:
            q = q.chunk(self.num_splits, dim=1)
            k = k.chunk(self.num_splits, dim=1)
            v = v.chunk(self.num_splits, dim=1)
        else:
            i = 0
            _q = []
            _k = []
            _v = []
            for h in self.shapes:
                _q.append(q[:, i:i+h, :, :])
                _k.append(k[:, i:i+h, :, :])
                _v.append(v[:, i:i+h, :, :])
                i = i+h
            q, k, v = _q, _k, _v


        attention = [natten2dqkrpb(_q, _k, _rpb, _kernel_size, _dilation) \
                     for _q,_k,_rpb,_kernel_size,_dilation in zip(q, k, self.rpb, self.kernel_sizes, self.dilations)]
        attention = [a.softmax(dim=-1) for a in attention]
        attention = [self.attn_drop(a) for a in attention]

        x = [natten2dav(_attn, _v, _k, _d) \
             for _attn, _v, _k, _d in zip(attention, v, self.kernel_sizes, self.dilations)]

        x = torch.cat(x, dim=1)
        x = x.permute(0, 2, 3, 1, 4).reshape(B, H, W, C)
        return self.proj_drop(self.proj(x))

class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, H):
        super().__init__()
        self.ln1 = nn.LayerNorm(H.bert_n_emb)
        self.ln2 = nn.LayerNorm(H.bert_n_emb)
        self.attn = CausalSelfAttention(H)
        
        h_params = H.items()
        # Save parameters of H
        with open("param.txt", "w") as file:
            for key, value in h_params:
                file.write(f"{key}: {value}\n")

        #self.attn = HydraNeighborhoodAttention(H)
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
