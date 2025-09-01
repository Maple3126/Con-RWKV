import os
import torch
import json
from tqdm import tqdm
from asr import SLAM_ASR
from dataset2 import MyDataset
from transformers import AutoModelForCausalLM
from argparse import ArgumentParser

def save_precomputed(cache_dir, sample_id, audio_embed, mask, label):
    os.makedirs(cache_dir, exist_ok=True)
    torch.save({
        "hubert": audio_embed,
        "mask": mask,
        "label": label,
    }, os.path.join(cache_dir, f"{sample_id}.pt"))

def run_cache(args):
    from asr import SLAM_ASR
    from transformers import AutoModelForCausalLM

    dataset = MyDataset(args.data_path, split= "train")

    model = SLAM_ASR(
        args=args,
        speech_encoder_model_id=args.speech_encoder_model_id,
        language_model=AutoModelForCausalLM.from_pretrained(args.language_model_id, trust_remote_code=True),
        train_mode="adapter",
        device="cuda",
    ).to("cuda")

    print("Start caching...")
    for i, (audio, text, sample_id) in enumerate(tqdm(dataset)):
        with torch.no_grad():
            hubert_output, _ = model.speech_encoder([audio])  # 抓出 HuBERT 原始輸出（未經 adapter）
            _, mask, label = model._prepare_input_embeds([audio], [text])  # 只為了拿 mask 與 label

        save_precomputed(args.cache_dir, sample_id, hubert_output.cpu(), mask.cpu(), label.cpu())
    print("Finished caching.")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)  # 放 dataset 檔案或資料資料夾
    # parser.add_argument("--tokenizer_path", type=str, default="RWKV/rwkv-6-world-1b6")
    parser.add_argument("--cache_dir", type=str, required=True)
    parser.add_argument("--speech_encoder_model_id", type=str, default="microsoft/wavlm-large")
    parser.add_argument("--language_model_id", type=str, default="RWKV/rwkv-6-world-1b6")
    parser.add_argument("--n_embd", type=int, required=True)
    args = parser.parse_args()

    run_cache(args)