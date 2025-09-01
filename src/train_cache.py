import os
import torch
import json
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset

class PrecomputedDataset(Dataset):
    def __init__(self, cache_dir):
        self.cache_dir = cache_dir
        self.files = [f for f in os.listdir(cache_dir) if f.endswith(".pt")]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = os.path.join(self.cache_dir, self.files[idx])
        data = torch.load(file_path)
        return data["hubert"].squeeze(0), data["mask"].squeeze(0), data["label"].squeeze(0)

def load_cached_dataloader(cache_dir, batch_size):
    dataset = PrecomputedDataset(cache_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    return loader

def run_train_from_cache(args, model):
    from pytorch_lightning import Trainer
    from pytorch_lightning.strategies import DeepSpeedStrategy

    train_loader = load_cached_dataloader(args.cache_dir, args.micro_bsz)
    val_loader = load_cached_dataloader(args.val_cache_dir, args.micro_bsz)

    print("[Info] Start training from cache...")
    args.precision = "bf16"  # ensure bf16 is passed if needed
    model.to("cuda")
    model.use_cached_forward = True  # <-- 告訴 forward 使用快取模式

    trainer = Trainer(
        max_epochs=args.epoch_count,
        accelerator="gpu",
        devices=args.devices,
        strategy=DeepSpeedStrategy(),
        precision=args.precision,
        logger=args.logger,
        callbacks=[],
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        log_every_n_steps=args.log_every_n_steps,
        enable_checkpointing=args.enable_checkpointing,
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=args.gradient_clip_val,
    )

    trainer.fit(model, train_loader, val_loader)
    print("[Info] Training finished.")

if __name__ == "__main__":
    from argparse import ArgumentParser
    from asr import SLAM_ASR
    from transformers import AutoModelForCausalLM

    parser = ArgumentParser()
    parser.add_argument("--cache_dir", type=str, required=True)
    parser.add_argument("--val_cache_dir", type=str, required=True)
    parser.add_argument("--epoch_count", type=int, default=10)
    parser.add_argument("--micro_bsz", type=int, default=2)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--logger", default=False)
    parser.add_argument("--check_val_every_n_epoch", type=int, default=1)
    parser.add_argument("--log_every_n_steps", type=int, default=50)
    parser.add_argument("--enable_checkpointing", action="store_true")
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    parser.add_argument("--gradient_clip_val", type=float, default=1.0)
    parser.add_argument("--speech_encoder_model_id", type=str, default="microsoft/wavlm-large")
    parser.add_argument("--language_model_id", type=str, default="RWKV/rwkv-6-world-1b6")
    args = parser.parse_args()

    model = SLAM_ASR(
        args=args,
        speech_encoder_model_id=args.speech_encoder_model_id,
        language_model=AutoModelForCausalLM.from_pretrained(args.language_model_id, trust_remote_code=True),
        train_mode="adapter",
        device="cuda",
    )

    run_train_from_cache(args, model)
