import os
import numpy as np
import struct
from RWKV_v5.src.binidx import MMapIndexedDataset

def generate_markov_batch(n_samples=10, n_tokens=1000, order=1, beta=1.0, seed=42):
    np.random.seed(seed)
    vocab = [0, 1]
    contexts = [tuple(map(int, format(i, f'0{order}b'))) for i in range(2 ** order)]
    transitions = {ctx: np.random.dirichlet([beta] * len(vocab)) for ctx in contexts}

    transitions = {}
    print("=== Transition Probabilities (from Dirichlet prior) ===")
    for ctx in contexts:
        probs = np.random.dirichlet([beta] * len(vocab))
        transitions[ctx] = probs
        # print(f"context {ctx} -> P(0)={probs[0]:.2f}, P(1)={probs[1]:.2f}")
    
    all_sequences = []
    sizes = []

    for _ in range(n_samples):
        seq = list(np.random.choice(vocab, size=order))
        for _ in range(n_tokens - order):
            ctx = tuple(seq[-order:])
            probs = transitions[ctx]
            next_token = np.random.choice(vocab, p=probs)
            seq.append(next_token)
        all_sequences.extend(seq)
        sizes.append(len(seq))

    return np.array(all_sequences, dtype=np.uint16), sizes

# def save_mmidx(data: np.ndarray, sizes: list, output_dir: str):
#     import os, struct
#     os.makedirs(output_dir, exist_ok=True)

#     with open(os.path.join(output_dir, "train.bin"), "wb") as f:
#         f.write(data.tobytes())

#     doc_idx = [0]  # 單一 doc
#     pointers = np.cumsum([0] + sizes[:-1])

#     with open(os.path.join(output_dir, "train.idx"), "wb") as f:
#         f.write(b"MMIDIDX\x00\x00")  # ✅ MMIDIDX header
#         f.write(struct.pack("<Q", 1))  # version
#         f.write(struct.pack("<B", 8))  # dtype code: 8 = uint16
#         f.write(struct.pack("<Q", len(sizes)))  # # samples
#         f.write(struct.pack("<Q", len(doc_idx)))  # # docs
#         f.write(np.array(sizes, dtype=np.int32).tobytes())  # sample sizes
#         f.write(np.array(pointers, dtype=np.int64).tobytes())  # sample offsets
#         f.write(np.array(doc_idx, dtype=np.int64).tobytes())  # doc idx

#     print(f"✅ 寫入 MMIDIDX 格式，共 {len(data)} tokens, {len(sizes)} samples")
#     # 呼叫
#     magic = find_magic_prime(len(tokens), 512)
#     print("✅ 合法 magic_prime =", magic)

def save_mmidx(data: np.ndarray, sizes: list, output_dir='data/markov-binidx'):  #########  run
    os.makedirs(output_dir, exist_ok=True)

    # --- 寫 bin ---
    bin_path = os.path.join(output_dir, 'train.bin')
    with open(bin_path, 'wb') as f:
        f.write(data.tobytes())

    # --- 寫 idx ---
    idx_path = os.path.join(output_dir, 'train.idx')
    with MMapIndexedDataset.Index.writer(idx_path, dtype=np.uint16) as writer:
        writer.write(sizes, doc_idx=[0])  # sizes: 每個 sample 的長度 (例如 512)

    print(f"✅ 成功寫入 MMIDIDX 格式，共 {len(data)} tokens")

# def save_binidx(data, output_dir='data/markov-binidx'):
#     os.makedirs(output_dir, exist_ok=True)

#     # Save train.bin
#     with open(os.path.join(output_dir, 'train.bin'), 'wb') as f:
#         f.write(data.tobytes())

#     # Save train.idx (32 bytes only)
#     idx_path = os.path.join(output_dir, 'train.idx')
#     with open(idx_path, 'wb') as f:
#         f.write(struct.pack('<8sQQQ', b'RWKVIDX\x00', 1, 0, len(data)))

#     size = os.path.getsize(idx_path)
#     assert size == 32, f"train.idx 應為 32 bytes，但實際為 {size}"
#     print(f"[✅] 成功寫入 train.idx ({size} bytes)")
#     print(f"[✅] 共 {len(data)} 個 tokens 可用於訓練")


def is_prime(n):
    if n < 2: return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0: return False
    return True

def find_magic_prime(total_tokens, ctx_len):
    slot = total_tokens // ctx_len
    for p in reversed(range(1, slot + 1)):
        if is_prime(p) and p % 3 == 2 and p / slot > 0.9:
            return p
    raise ValueError("找不到合法的 magic_prime")


# ==== 執行區 ====

if __name__ == '__main__':
    tokens, sizes = generate_markov_batch(n_samples=42, n_tokens=512, order=2, beta=1.0)
    save_mmidx(tokens, sizes, output_dir='data/markov-binidx')
    