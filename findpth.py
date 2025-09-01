import torch
import argparse

def check_model_arch(path):
    print(f"🔍 Loading: {path}")
    ckpt = torch.load(path, map_location='cpu')

    # 嘗試多種 key 可能性
    if 'model' in ckpt:
        state_dict = ckpt['model']
    elif 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    else:
        state_dict = ckpt

    # 嘗試找出 layer 數量
    layer_count = 0
    possible_prefixes = ['', 'language_model.', 'rwkv.', 'model.', 'backbone.']
    for prefix in possible_prefixes:
        att_keys = [k for k in state_dict if k.startswith(f"{prefix}blocks") and "att.time_state" in k]
        if att_keys:
            layer_count = len(att_keys)
            break

    print(f"🧱 Detected number of layers: {layer_count}")

    # 嘗試找 embedding size
    embd_keys = [k for k in state_dict if 'emb.weight' in k or 'embeddings.weight' in k]
    if embd_keys:
        embd_size = state_dict[embd_keys[0]].shape[1]
        print(f"📐 Detected embedding size (n_embd): {embd_size}")
    else:
        print("⚠️ Could not find embedding size key")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt", help="Path to .pth checkpoint file")
    args = parser.parse_args()
    check_model_arch(args.ckpt)