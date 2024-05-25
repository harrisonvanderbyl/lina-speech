from dataclasses import dataclass
from typing import Optional
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from model.attentive_rnn import AttentiveRNN
from model.crossatt import BlindCrossAttention, CrossAttention
from model.base_blocks import MixingBlock, SwiGLU
from einops import rearrange

from torch.utils.cpp_extension import load
from fla.ops.rwkv6.recurrent_fuse import FusedRecurrentRWKV6Function
from fla.ops.rwkv6.chunk import chunk_rwkv6
#adapted from https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v5/src/model.py

def __noop(ob):
    return ob


MyFunction = __noop




def RUN_CUDA_RWKV6(B, T, C, H, r, k, v, w, u):
    return WKV_6.apply(B, T, C, H, r, k, v, w, u)

class ctxx():
    def save_for_backward(*args):
        pass
    
class RWKV_TMix_x060(nn.Module):
    def __init__(self, d_model:int, n_head:int, layer_id:int, n_layer:int, head_size_divisor: int=8):
        super().__init__()
        self.layer_id = layer_id

        self.head_size = d_model // n_head
        self.head_size_divisor = head_size_divisor
        self.n_head = n_head
        assert d_model % self.n_head == 0
        global wkv6_cuda

        with torch.no_grad():
            ratio_0_to_1 = layer_id / (n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, d_model)
            for i in range(d_model):
                ddd[0, 0, i] = i / d_model

            # fancy time_mix
            self.time_maa_x = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_w = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_v = nn.Parameter(
                1.0 - (torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1)
            )
            self.time_maa_r = nn.Parameter(
                1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0)
            )
            self.time_maa_g = nn.Parameter(
                1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0)
            )

            TIME_MIX_EXTRA_DIM = 32  # generate TIME_MIX for w,k,v,r,g
            self.time_maa_w1 = nn.Parameter(
                torch.zeros(d_model, TIME_MIX_EXTRA_DIM * 5)
            )
            self.time_maa_w2 = nn.Parameter(
                torch.zeros(5, TIME_MIX_EXTRA_DIM, d_model).uniform_(-0.01, 0.01)
            )

            # fancy time_decay
            decay_speed = torch.ones(d_model)
            for n in range(d_model):
                decay_speed[n] = -6 + 5 * (n / (d_model - 1)) ** (
                    0.7 + 1.3 * ratio_0_to_1
                )
            self.time_decay = nn.Parameter(decay_speed.reshape(1, 1, d_model))

            TIME_DECAY_EXTRA_DIM = 64
            self.time_decay_w1 = nn.Parameter(
                torch.zeros(d_model, TIME_DECAY_EXTRA_DIM)
            )
            self.time_decay_w2 = nn.Parameter(
                torch.zeros(TIME_DECAY_EXTRA_DIM, d_model).uniform_(-0.01, 0.01)
            )

            tmp = torch.zeros(d_model)
            for n in range(d_model):
                zigzag = ((n + 1) % 3 - 1) * 0.1
                tmp[n] = ratio_0_to_1 * (1 - (n / (d_model - 1))) + zigzag

            self.time_faaaa = nn.Parameter(tmp.reshape(self.n_head, self.head_size))

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.receptance = nn.Linear(d_model, d_model, bias=False)
        self.key = nn.Linear(d_model, d_model, bias=False)

        self.value = nn.Linear(d_model, d_model, bias=False)
        self.output = nn.Linear(d_model, d_model, bias=False)
        self.gate = nn.Linear(d_model, d_model, bias=False)
        self.ln_x = nn.GroupNorm(
            self.n_head, d_model, eps=(1e-5) * (self.head_size_divisor**2)
        )
        self.kv_state = None
    

    @MyFunction
    def jit_func(self, x):
        B, T, C = x.size()

        if self.inference:
            state = self.state
            if state is None:
                state = torch.zeros_like(x)
        else:
            state = self.time_shift(x)

        xx = state - x

        xxx = x + xx * self.time_maa_x
        xxx = torch.tanh(xxx @ self.time_maa_w1).view(B * T, 5, -1).transpose(0, 1)
        xxx = torch.bmm(xxx, self.time_maa_w2).view(5, B, T, -1)
        mw, mk, mv, mr, mg = xxx.unbind(dim=0)

        xw = x + xx * (self.time_maa_w + mw)
        xk = x + xx * (self.time_maa_k + mk)
        xv = x + xx * (self.time_maa_v + mv)
        xr = x + xx * (self.time_maa_r + mr)
        xg = x + xx * (self.time_maa_g + mg)

        r = self.receptance(xr)
        k = self.key(xk)
        v = self.value(xv)
        g = F.silu(self.gate(xg))

        ww = torch.tanh(xw @ self.time_decay_w1) @ self.time_decay_w2
        w = self.time_decay + ww
        
        if self.inference:
            self.state = x

        return r, k, v, g, w

    @MyFunction
    def jit_func_2(self, x, g):
        B, T, C = x.size()
        x = x.view(B * T, C)

        x = self.ln_x(x).view(B, T, C)
        x = self.output(x * g)
        return x

    def forward(self, x):
        B, T, C = x.size()
        H = self.n_head
        K = self.head_size

        r, k, v, g, w = self.jit_func(x)
        if self.inference:
            x,self.kv_state = FusedRecurrentRWKV6Function.forward(ctxx(),r.view(B,T,H,K).transpose(1,2),k.view(B,T,H,K).transpose(1,2),v.view(B,T,H,K).transpose(1,2),w.view(B,T,H,K).transpose(2,1).exp().neg(),self.time_faaaa.view(H,K),1.0,self.kv_state,True,0)
        else:
            x = RUN_CUDA_RWKV6(B, T, C, H, r, k, v, w.bfloat16(), u=self.time_faaaa.bfloat16())
        x = x.view(B , T, C)
        return self.jit_func_2(x, g)


class RWKV_CMix_x060(nn.Module):
    def __init__(self, d_model:int, layer_id:int, n_layer:int):
        super().__init__()
        self.layer_id = layer_id
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        dim_ffn = int((d_model * 3.5) // 32 * 32)

        with torch.no_grad():  # fancy init of time_mix
            ratio_1_to_almost0 = 1.0 - (layer_id / n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, d_model)
            for i in range(d_model):
                ddd[0, 0, i] = i / d_model
            self.time_maa_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_r = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))

        self.key = nn.Linear(d_model, dim_ffn, bias=False)
        self.receptance = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(dim_ffn, d_model, bias=False)
        self.inference = False
        self.state = None

    @MyFunction
    def forward(self, x):
        if self.inference:
            state = self.state
            if state is None:
                state = torch.zeros_like(x)
        else:
            state = self.time_shift(x)

        xx =  state - x
        xk = x + xx * self.time_maa_k
        xr = x + xx * self.time_maa_r

        k = self.key(xk)
        k = torch.relu(k) ** 2
        kv = self.value(k)
        if self.inference:
            self.state = x
        return torch.sigmoid(self.receptance(xr)) * kv



def exists(x):
    return x is not None

class AttentiveRWKV6(AttentiveRNN):
    def __init__(
        self,
        d_model:int,
        n_layer:int,
        d_context:int,
        heads:int,
        dropout_att:float=0.0,
        d_blind:int=128,
        blind=False,
        light=False,
    ):
        super().__init__()
        block = lambda dim, heads, lay_i, n_lay: MixingBlock(lambda: RWKV_TMix_x060(dim, heads, lay_i, n_lay),
                                                   lambda: SwiGLU(dim) if light else RWKV_CMix_x060(dim, lay_i, n_lay),
                                                   lambda: nn.LayerNorm(dim),)
        self.encoder = nn.Sequential(*[block(d_model, heads, i, n_layer) for i in range(n_layer)])
        self.decoder = nn.Sequential(*[block(d_model, heads, i, n_layer) for i in range(n_layer)])
        self.cross_att = (
            BlindCrossAttention(
                d_model, d_context, d_model, heads, block(d_blind, 1, 0, 2), dropout_att, pos_dim=d_blind,
            )
            if blind
            else CrossAttention(d_model, d_context, d_model, heads, dropout_att)
        )

    def forward(self, x, ctx, x_mask=None, ctx_mask=None):
        if exists(x_mask) and exists(ctx_mask):
            mask = rearrange(x_mask, "b i -> b i 1") * rearrange(
                ctx_mask, "b j -> b 1 j"
            )
            mask = rearrange(mask, "b i j -> b 1 i j")
        else:
            mask = None
        y = torch.utils.checkpoint.checkpoint_sequential(self.encoder, len(self.encoder), x, use_reentrant=False)
        v, att = self.cross_att(y, ctx, mask=mask)
        y = torch.utils.checkpoint.checkpoint_sequential(self.decoder, len(self.decoder), y + v, use_reentrant=False)
 
        return y, att

    def init_state(self, max_seqlen=1000):
        for be, bd in zip(self.encoder, self.decoder):
            be.tmix.inference, bd.tmix.inference = True, True
            be.cmix.inference, bd.cmix.inference = True, True
            be.tmix.state, bd.tmix.state = None, None
            be.cmix.state, bd.cmix.state = None, None
            be.tmix.kv_state, bd.tmix.kv_state = None, None
        self.cross_att.pos_net.tmix.inference = True
        self.cross_att.pos_net.cmix.inference = True
        self.cross_att.pos_net.tmix.state = None
        self.cross_att.pos_net.cmix.state = None
        self.cross_att.pos_net.tmix.kv_state = None
        
    def set_state(self, state):
        for be, bd in zip(self.encoder, self.decoder):
            be.tmix.inference, bd.tmix.inference = True, True
            be.cmix.inference, bd.cmix.inference = True, True
            be.tmix.state, bd.tmix.state = None, None
            be.cmix.state, bd.cmix.state = None, None
            be.tmix.kv_state, bd.tmix.kv_state = None, None
            
        self.cross_att.pos_net.tmix.inference = True
        self.cross_att.pos_net.cmix.inference = True
        
        self.cross_att.pos_net.tmix.state = state[0]
        self.cross_att.pos_net.cmix.state = state[1]
        self.cross_att.pos_net.tmix.kv_state = state[2]
        for num, i in enumerate(self.encoder):
            i.tmix.state = state[3][num][0]
            i.cmix.state = state[3][num][1]
            i.tmix.kv_state = state[3][num][2]
        for num, i in enumerate(self.decoder):
            i.tmix.state =  state[4][num][0]
            i.cmix.state = state[4][num][1]
            i.tmix.kv_state = state[4][num][2]
        
    def get_state(self):
        return [self.cross_att.pos_net.tmix.state, 
                self.cross_att.pos_net.cmix.state, 
                self.cross_att.pos_net.tmix.kv_state, 
                [[i.tmix.state, i.cmix.state, i.tmix.kv_state] for i in self.encoder], 
                [[i.tmix.state, i.cmix.state, i.tmix.kv_state] for i in self.decoder]]


    def step(self, y_embd, x_enc, time_step):
        y_embd = self.encoder(y_embd)
        v, att = self.cross_att(y_embd, x_enc, time_step=time_step)
        y_embd = y_embd + v
        y_embd = self.decoder(y_embd)
        return y_embd, att
