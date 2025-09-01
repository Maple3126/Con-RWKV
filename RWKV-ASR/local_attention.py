import math
import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat, pack, unpack

class RoPEPositionEmbedding(nn.Module):
    def __init__(self, dim: int, max_len: int = 512, base: int = 10000) -> None:
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_len = max_len

        theta = 1 / (base ** (torch.arange(0, dim, 2).float() / dim))
        position_ids = torch.arange(0, max_len)
        # 預計算每一個位置的 position_matrix
        position_matrix = position_ids.unsqueeze(1) * theta.unsqueeze(0)  # [max_len, dim//2]
        cos = torch.cos(position_matrix)  # [max_len, dim//2]
        sin = torch.sin(position_matrix)
        # 為了與輸入的dim對齊，重複擴展
        cos = cos.repeat_interleave(2, dim=-1)  # [max_len, dim]
        sin = sin.repeat_interleave(2, dim=-1)
        self.register_buffer('cos', cos) # [max_len, dim]
        self.register_buffer('sin', sin)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, L, d] 或 [L, d]，d==dim
        """
        L = x.shape[-2]
        cos, sin = self.cos[:L], self.sin[:L]  # 直接索引已預計算好範圍
        if x.dim() == 3:
            cos = cos.unsqueeze(0)
            sin = sin.unsqueeze(0)
        # 90度旋轉
        x_rot = torch.empty_like(x)
        x_rot[..., 0::2] = -x[..., 1::2]
        x_rot[..., 1::2] = x[..., 0::2]
        return x * cos + x_rot * sin



# 常數
TOKEN_SELF_ATTN_VALUE = -5e4

# helper functions
def exists(val):
    return val is not None

def default(value, d):
    return d if not exists(value) else value

def to(t):
    return {'device': t.device, 'dtype': t.dtype}

def max_neg_value(tensor):
    return -torch.finfo(tensor.dtype).max

def l2norm(tensor):
    dtype = tensor.dtype
    normed = F.normalize(tensor, dim=-1)
    return normed.type(dtype)

def pad_to_multiple(tensor, multiple, dim=-1, value=0):
    seqlen = tensor.shape[dim]
    m = seqlen / multiple
    if m.is_integer():
        return False, tensor
    remainder = math.ceil(m) * multiple - seqlen
    pad_offset = (0,) * (-1 - dim) * 2
    return True, F.pad(tensor, (*pad_offset, 0, remainder), value=value)

def look_around(x, backward=1, forward=0, pad_value=0, dim=2):
    dims = (len(x.shape) - dim) * (0, 0)
    padded_x = F.pad(x, (*dims, backward, forward), value=pad_value)
    tensors = padded_x.unfold(1, forward + backward + 1, 1).contiguous()
    return tensors.movedim(-1, dim).flatten(dim, dim + 1)

# main class
class LocalAttention(nn.Module):
    def __init__(
        self,
        window_size,
        attn_dim=None,
        causal=True,
        look_backward=1,
        look_forward=None,
        dropout=0.1,
        shared_qk=False,
        rel_pos_emb_config=None,
        dim=None,
        autopad=False,
        exact_windowsize=False,
        scale=None,
        use_rotary_pos_emb=False,
        use_xpos=False,
        xpos_scale_base=None,
        max_seq_len=2000  # 可根據需求調整最大序列長度
    ):
        super().__init__()
        look_forward = default(look_forward, 0 if causal else 1)
        assert not (causal and look_forward > 0), 'you cannot look forward if causal'

        self.scale = scale
        self.window_size = window_size
        self.autopad = autopad
        self.exact_windowsize = exact_windowsize
        self.causal = causal
        self.look_backward = look_backward
        self.look_forward = look_forward
        self.dropout = nn.Dropout(dropout)
        self.shared_qk = shared_qk
        self.use_xpos = use_xpos
        
        @property
        def weights(self):
            return {
                'wq': self.wq.weight,
                'wk': self.wk.weight,
                'wv': self.wv.weight
            }

        @property
        def biases(self):
            return {
                'wq': self.wq.bias if self.wq.bias is not None else None,
                'wk': self.wk.bias if self.wk.bias is not None else None,
                'wv': self.wv.bias if self.wv.bias is not None else None
            }

        # 投影層：假設 attn_dim 即為模型的 hidden dimension
        # self.wq = nn.Linear(attn_dim, attn_dim)
        # self.wk = nn.Linear(attn_dim, attn_dim)
        # self.wv = nn.Linear(attn_dim, attn_dim)
        # self.wq = nn.Conv1d(attn_dim, attn_dim, kernel_size=window_size, padding='same', groups=attn_dim, bias=True)
        # self.wk = nn.Conv1d(attn_dim, attn_dim, kernel_size=window_size, padding='same', groups=attn_dim, bias=True)
        # self.wv = nn.Conv1d(attn_dim, attn_dim, kernel_size=window_size, padding='same', groups=attn_dim, bias=True)

        # padding_left = (window_size - 1) // 2
        # padding_right = window_size // 2
        # self.pad = nn.ConstantPad1d((padding_left, padding_right), 0.0)
        # self.wq = nn.Conv1d(attn_dim, attn_dim, kernel_size=window_size, padding=0, groups=attn_dim, bias=False)
        # self.wk = nn.Conv1d(attn_dim, attn_dim, kernel_size=window_size, padding=0, groups=attn_dim, bias=False)
        # self.wv = nn.Conv1d(attn_dim, attn_dim, kernel_size=window_size, padding=0, groups=attn_dim, bias=False)

        # 卷積層：左填充
        self.left_pad = nn.ConstantPad1d((window_size - 1, 0), 0.0)  # 只在左側填充
        self.wq = nn.Conv1d(attn_dim, attn_dim, kernel_size=window_size, padding=0, groups=attn_dim, bias=False)
        self.wk = nn.Conv1d(attn_dim, attn_dim, kernel_size=window_size, padding=0, groups=attn_dim, bias=False)
        self.wv = nn.Conv1d(attn_dim, attn_dim, kernel_size=window_size, padding=0, groups=attn_dim, bias=False)


        # 如果使用 RoPE 位置編碼，建立對應模組（對 query 與 key 均使用）
        self.use_rotary_pos_emb = use_rotary_pos_emb
        if self.use_rotary_pos_emb:
            self.rope = RoPEPositionEmbedding(dim=attn_dim, max_len=max_seq_len)

    def forward(
        self,
        q, k, v,
        mask=None,
        input_mask=None,
        attn_bias=None,
        window_size=None,
        pad_lengths=None,
    ):
        mask = default(mask, input_mask)
        # print("mask", mask.shape, mask[0])

        assert not (exists(window_size) and not self.use_xpos), 'cannot perform window size extrapolation if xpos is not turned on'

        # 投影得到 q, k, v
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()
        # print(f'q shape: {q.shape}, k shape: {k.shape}, v shape: {v.shape}')
        q = self.left_pad(q)
        k = self.left_pad(k)
        v = self.left_pad(v)
        # print(f'q shape: {q.shape}, k shape: {k.shape}, v shape: {v.shape}')
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        # print(f'q shape: {q.shape}, k shape: {k.shape}, v shape: {v.shape}')
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        # print(f'q shape: {q.shape}, k shape: {k.shape}, v shape: {v.shape}')
        # print("q", q.shape, q[0])

        # 使用 RoPE 位置編碼（僅對 q, k）
        if self.use_rotary_pos_emb:
            q = self.rope(q)
            k = self.rope(k)

        shape, autopad, pad_value, window_size, causal, look_backward, look_forward, shared_qk = q.shape, self.autopad, 0, default(window_size, self.window_size), self.causal, self.look_backward, self.look_forward, self.shared_qk

        # 使用 einops pack 處理
        (q, packed_shape), (k, _), (v, _) = map(lambda t: pack([t], '* n d'), (q, k, v))

        # 自動 padding
        if autopad:
            orig_seq_len = q.shape[1]
            (needed_pad, q), (_, k), (_, v) = map(lambda t: pad_to_multiple(t, self.window_size, dim=-2), (q, k, v))
            if exists(mask):
                _, mask_pad = pad_to_multiple(mask, self.window_size, dim=-1, value=True)# False

        # print("after autopad: ")
        # print("q: ", q.shape, q[0])
        # print("mask_pad: ", mask_pad.shape, mask_pad[0])

        b, n, dim_head, device, dtype = *q.shape, q.device, q.dtype
        scale = default(self.scale, dim_head ** -0.5)

        assert (n % window_size) == 0, f'sequence length {n} must be divisible by window size {window_size} for local attention'
        windows = n // window_size

        if shared_qk:
            k = l2norm(k)

        seq = torch.arange(n, device=device)
        b_t = rearrange(seq, '(w n) -> 1 w n', w=windows, n=window_size)

        # bucketing：重整張量 shape
        bq, bk, bv = map(lambda t: rearrange(t, 'b (w n) d -> b w n d', w=windows), (q, k, v))
        bq = bq * scale
        # print("bq", bq[0])

        look_around_kwargs = dict(
            backward=look_backward,
            forward=look_forward,
            pad_value=pad_value
        )
        # print("bk", bk.shape, bk[0])
        # print("bv", bk.shape, bv[0])
        # print("backward", look_backward)
        bk = look_around(bk, **look_around_kwargs)
        bv = look_around(bv, **look_around_kwargs)
        # print("After look_around")
        # print("bk: ", bk[0])
        # print("bv: ", bv[0])

        # 計算位置，用於 masking
        bq_t = b_t
        bq_k = look_around(b_t, **look_around_kwargs)
        bq_t = rearrange(bq_t, '... i -> ... i 1')
        bq_k = rearrange(bq_k, '... j -> ... 1 j')
        pad_mask = bq_k == pad_value

        sim = einsum('b h i e, b h j e -> b h i j', bq, bk)
        # print("after einsum: ")
        # print("sim: ", sim[0])

        if exists(attn_bias):
            heads = attn_bias.shape[0]
            assert (b % heads) == 0
            attn_bias = repeat(attn_bias, 'h i j -> (b h) 1 i j', b=b // heads)
            sim = sim + attn_bias

        # mask_value = max_neg_value(sim)
        mask_value = -1e4

        if shared_qk:
            self_mask = bq_t == bq_k
            sim = sim.masked_fill(self_mask, TOKEN_SELF_ATTN_VALUE)
            del self_mask

        if causal:
            causal_mask = bq_t < bq_k
            if self.exact_windowsize:
                max_causal_window_size = (self.window_size * self.look_backward)
                causal_mask = causal_mask | (bq_t > (bq_k + max_causal_window_size))
            sim = sim.masked_fill(causal_mask, mask_value)
            # print("causal: ")
            # print("sim: ", sim[0])

            del causal_mask

        if not causal and self.exact_windowsize:
            max_backward_window_size = (self.window_size * self.look_backward)
            max_forward_window_size = (self.window_size * self.look_forward)
            window_mask = ((bq_k - max_forward_window_size) > bq_t) | (bq_t > (bq_k + max_backward_window_size)) | pad_mask
            sim = sim.masked_fill(window_mask, mask_value)
        else:
            sim = sim.masked_fill(pad_mask, mask_value)

        if exists(mask):
            batch = mask.shape[0]
            assert (b % batch) == 0
            h = b // mask.shape[0]
            if autopad:
                _, mask_pad = pad_to_multiple(mask, window_size, dim=-1, value=False)
            mask_pad = ~mask_pad.bool()
            # print("mask_pad", mask_pad.shape, mask_pad[0])
            mask_pad = rearrange(mask_pad, '... (w n) -> (...) w n', w=windows, n=window_size)
            mask_pad = look_around(mask_pad, **{**look_around_kwargs, 'pad_value': False})
            mask_pad = rearrange(mask_pad, '... j -> ... 1 j')
            mask_pad = repeat(mask_pad, 'b ... -> (b h) ...', h=h)
            sim = sim.masked_fill(~mask_pad, mask_value)
            # print("mask: ")
            # print("sim: ", sim[0])
            del mask_pad
            

        attn = sim.softmax(dim=-1)
        # print("attn", attn[0])
        attn = self.dropout(attn)
        
        # print("bv", bv[0])
        out = einsum('b h i j, b h j e -> b h i e', attn, bv)
        out = rearrange(out, 'b w n d -> b (w n) d')
        # print("After attn: ")
        # print("out: ", out.shape, out[0])

        if autopad:
            out = out[:, :orig_seq_len, :]
            # print("After autopad: ")
            # print("out: ", out.shape, out[0])

        out, *_ = unpack(out, packed_shape, '* n d')
        

        if exists(mask):
            # print("mask", mask.shape, mask[0])
            out = out.masked_fill(mask.unsqueeze(-1), 0.)
            # print("After mask: ")
            # print("out: ", out.shape, out[0])
            del mask

        return out
    @property
    def weights(self):
        return {
            'wq': self.wq.weight,
            'wk': self.wk.weight,
            'wv': self.wv.weight
        }

    @property
    def biases(self):
        return {
            'wq': self.wq.bias if self.wq.bias is not None else None,
            'wk': self.wk.bias if self.wk.bias is not None else None,
            'wv': self.wv.bias if self.wv.bias is not None else None
        }
