import os
os.environ["RWKV_JIT_ON"] = "0"
os.environ["RWKV_HEAD_SIZE_A"] = "64"

import types, torch
from src.model import RWKV          # ← 你的 RWKV 類別檔案
                               #   如果檔名不同請改 import 路徑

# >>> 把「跟訓練時完全一樣」的超參數塞進 args
import types

args = types.SimpleNamespace(
    # === 跟 training script 完全一致的部分 ===
    my_pos_emb = 0,
    n_layer = 2,
    n_embd = 768,
    vocab_size = 2,
    ctx_len = 512,
    head_size_a = 64,
    head_size_divisor = 8,
    my_pile_stage = 1,
    epoch_count = 1,
    epoch_begin = 0,
    epoch_save = 1,
    weight_decay = 0.0,
    pre_ffn = 0,
    head_qk = 0,
    my_exit_tokens = 100000,
    magic_prime = 179,
    lr_init = 1e-5,
    lr_final = 1e-5,
    warmup_steps = 10,
    beta1 = 0.9,
    beta2 = 0.99,
    adam_eps = 1e-8,
    my_pile_edecay = 0,
    num_nodes = 1,
    micro_bsz = 1,
    accelerator = "cpu",
    devices = "1",
    precision = "bf16",
    strategy = "deepspeed_stage_2",
    grad_cp = 1,
    my_testing = "x060",
    my_exit = 99999999,
    
    # === ArgumentParser 裡的預設值（你沒設定但可能需要） ===
    op = "train",
    load_model = "",
    wandb = "",
    proj_dir = "out",
    random_seed = -1,
    data_type = "binidx",
    epoch_steps = 1000,
    dim_att = 0,
    dim_ffn = 0,
    tiny_att_dim = 0,
    tiny_att_layer = -999,
    dropout = 0.0,
    weight_decay_final = -1,
    my_pile_version = 1,
    my_pile_shift = 0,
    layerwise_lr = 1,
    ds_bucket_mb = 200,
    my_sample_len = 0,
    my_ffn_shift = 1,
    my_att_shift = 1,
    load_partial = 0,
    my_qa_mask = 0,
    my_random_steps = 0,
    train_type = "none",
    file_path = "none",
    loss_mask = False,
    chunk_ctx = 512,

    # === LoRA 預設關閉 ===
    emb = False,
    lora = False,
    lora_load = "",
    lora_r = 8,
    lora_alpha = 32,
    lora_dropout = 0.01,
    lora_parts = "att,ln,time",

    # === LISA 預設關閉 ===
    LISA = False,
    lisa_r = 2,
    lisa_k = 100,

    # === PISSA 預設關閉 ===
    PISSA = False,
    svd_niter = 4,
    pissa_load = "",
    pissa_init = "",

    # === 量化相關 ===
    quant = "none",

    # === dataset 預設 ===
    dataload = "get",

    # === conv 預設關閉 ===
    conv = False,
    conv_kernel = 5,
    conv_dummy_scale = 0.0
)
# RWKV 內部還會用到 dim_att / dim_ffn，會在 __init__ 裡自動算

# 1. 先建「空殼」模型
model = RWKV(args).bfloat16().cuda()   # 或 .to('cpu') 皆可

# 2. 把檔案 load 進來
model.load_state_dict(
    torch.load('out/L2-D768-x060/rwkv-init.pth', map_location='cuda')
)
model.eval()

seq = [0, 1, 1, 0, 1, 0]        # 任何你想測的序列
x   = torch.tensor(seq, dtype=torch.long, device='cuda').unsqueeze(0)  # [1,T]

with torch.no_grad():
    logits = model(x)           # [1,T,2]
    probs  = torch.softmax(logits, dim=-1)

# 機率預測：最後一個位置對 token=1 的機率
p_model = probs[0, -1, 1].item()
print(f"Model p(xt+1=1 | x1..t) = {p_model:.4f}")

order = 1        # 你現在練的是 1-order Markov
beta  = 1.0

# 取最後 k=order 個 token 當作 context
ctx = seq[-order:]

# 數序列內所有跟這個 context 相同的 transition
n, n1 = 0, 0
for i in range(order, len(seq)):
    if seq[i-order:i] == ctx:
        n  += 1
        n1 += (seq[i] == 1)

p_laplace = (n1 + beta) / (n + 2*beta)
print(f"Laplace estimator       = {p_laplace:.4f}")

print(f"Abs diff = {abs(p_model - p_laplace):.4e}")