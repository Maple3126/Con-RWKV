########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import json, math, random, os, sys
import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset
from pytorch_lightning.utilities import rank_zero_info
# from .binidx import MMapIndexedDataset
# from .utils import MaybeIsPrime
from rwkv.utils import PIPELINE
import librosa
pipeline = PIPELINE('rwkv6', "rwkv_vocab_v20230424")
import glob
class MyDataset(Dataset):
    def __init__(self, args, split="train"):
        if split == "train":
            self.data_dirs =  [
                "/home/TEDDATA/tedtrain"
                # "/home/LiSpeech100c/train-clean-100",
                # "/home/train360c",
                # "/home/train500"
            ]
            # self.data_dir = "/home/testclean100" 
        elif split == "valid":
            self.data_dirs = ["/home/TEDDATA/teddev"]   #/home/devclean100/dev-clean
        elif split == "test_c":
            self.data_dirs = ["/home/TEDDATA/tedtest"]    #/home/testclean100
        elif split == "test_oth":
            self.data_dirs = [ "/home/testother100" ]      #/home/testother100
        else:
            raise ValueError("Invalid split. Use 'train' or 'valid'.")
        self.samples = []

        for data_dir in self.data_dirs:
            for speaker in os.listdir(data_dir):
                spk_dir = os.path.join(data_dir, speaker)
                if not os.path.isdir(spk_dir):
                    continue
                for chapter in os.listdir(spk_dir):
                    ch_dir = os.path.join(spk_dir, chapter)
                    if not os.path.isdir(ch_dir):
                        continue
                    # 嘗試找到任一個 *.trans.txt 檔案
                    trans_files = glob.glob(os.path.join(ch_dir, "*.trans.txt"))
                    if not trans_files:
                        print(f"❌ No transcript file found in {ch_dir}")
                        continue
                    trans_file = trans_files[0]  # 拿第一個 transcript 檔案
                    
                    with open(trans_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            utt_id, *text = line.strip().split()
                            wav_path = os.path.join(ch_dir, f"{utt_id}.wav")
                            if os.path.exists(wav_path):
                                self.samples.append((wav_path, ' '.join(text)))
        print(f"Loaded {len(self.samples)} samples from {self.data_dirs}")


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        wav_path, text = self.samples[idx]
        audio, sr = torchaudio.load(wav_path)
        # sample_id = os.path.basename(wav_path).replace(".wav", "")  # 加上這行
        
        return (audio.squeeze(0).numpy(), text.lower())  #有改

         
        