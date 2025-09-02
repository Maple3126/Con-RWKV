from src.binidx import MMapIndexedDataset
data = MMapIndexedDataset("/home/RWKVICL/RWKVLM/data/markov-binidx/train")
print(f"總共有 {len(data)} 筆 sample")
print(f"第一筆的 token 數量：{len(data[0])}")
print(f"全部 token 數量：約 {sum(data.sizes)}")
for i in range(10):
    tokens = data[i]  # 每個 sample 是一段 token list (通常是 numpy array)
    print(f"Sample {i}: {tokens.tolist()}")