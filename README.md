## RWKV-ASR

This project is based on the SLAM-ASR framework, with Con-RWKV integrated as the final output language model (LLM). 
The ASR task can be completed by running the demo-state-tuning.sh script, which contains the core setup and execution flow.

### ENV
```bash
conda create -n rwkv python=3.10
conda activate rwkv
pip install -r requirements.txt
```

### Training
1. Download RWKV-6-World model files from one of the following links. We used the 3B model in our experiments at most, i.e. RWKV-x060-World-3B-v2.1-20240417-ctx4096.pth.

- [Hugging Face](https://huggingface.co/BlinkDL/rwkv-6-world/tree/main) 
- [Hf Mirror (CN)](https://hf-mirror.com/BlinkDL/rwkv-6-world/tree/main) 
- [Modelscope](https://modelscope.cn/models/Blink_DL/rwkv-6-world/files)

2. Set ```OP=train``` for training and ```load_model=path/to/your/model/```. Based on the model you are using, please modify the ```n_embed``` and ```n_layer``` parameters in the demo-state-tuning.sh script as follows:
   | Model Parameter | $n_{\text{layer}}$ | $n_{\text{embd}}$ |
|-----------------|--------------------|--------------------|
| 0.4B(rwkv-5)            | 24                 | 1024               |
| 1.6B            | 24                 | 2048               |
| 3B              | 32                 | 2560               |
| 7B              | 32                 | 4096               |
| 14B             | 61                 | 4096               |

Other parameters for training:
|   parameter       | description  |
| --------- | ---- |
| conv | add convolution module or not | 
| kernel size | conv kernel size (suggest 5) | 
| micro_bsz | batch size for each device | 
| epoch_steps | num of steps in 1 epoch. please modified as (dataset size / real batch size) | 
| device | num of GPU for training |  

Pick the encoder in ```train.py```, HuBERT large-ls960 is specifically fine-tuned for LibriSpeech.

3. The script will overwrite the .pth file in ```output/```. Make sure to save the needed .pth model files under this path to other dir before the training.
4. run ```sh demo/demo-state-tuning.sh``` to start the training process.

### Evaluation

Follow the instruction in Training, but modify ```OP=eval``` in ```demo/demo-state-tuning.sh```. 
The trained model in ```output/``` will be used to calculate the WER of the model in ```output/``` on the clean test set and the other test set of Librispeech.
The data can be modified in dataset2.py, for both train and eval.

### Most of the above procedures are referenced from AGENDD / RWKV-ASR, and the details can be consulted in that repository.

## RWKV-ICL
1. Use ```datanew.py``` to generate binary toy data following a Dirichlet distribution. Each subsequent token is generated based on the preceding k tokens, where k can be set by modifying the ```if_main_``` section.
The initial k tokens, however, are sampled from a uniform distribution.
2.As the experiments are based on RWKV-6, the ```demo-training-run.sh``` script is executed within the ```RWKVLM/RWKV_v5``` directory.

The internal settings of the ```.sh``` file are mostly similar to those in RWKV-ASR. The only point to note is that when switching to toy data with a different order, it is safer to manually clear the contents of ```data/markov-binidx.``` 
In addition, the convolution kernel size should be set to $\geq k+1$ for the loss to decrease significantly.
