#0.4B
# 24 1024


# 3B
load_model='RWKV-x060-World-3B-v2.1-20240417-ctx4096.pth' 
#7B
# load_model='RWKV-x060-World-7B-v2.1-20240507-ctx4096.pth'  

#model output dir
proj_dir='output'

# 3B
n_layer=32
n_embd=2560

# 7B
# n_layer=32
# n_embd=4096

micro_bsz=4
epoch_steps=18089
ctx_len=1024
device=1
epoch_save=2

OP="train"

QUANT='nf4' 

python train.py --load_model '/home/RWKV-ASR/RWKV-x060-World-3B-v2.1-20240417-ctx4096.pth' --devices 1 \
--proj_dir $proj_dir \
--data_type binidx --vocab_size 65536 --dropout 0 \
--ctx_len $ctx_len --epoch_steps 12000 --epoch_count 3 --epoch_begin 0 --epoch_save $epoch_save --micro_bsz $micro_bsz \
--n_layer 32 --n_embd 2560 \
--pre_ffn 0 --head_qk 0 --lr_init 5e-4 --lr_final 1e-4 --warmup_steps 50 --beta1 0.9 --beta2 0.99 --adam_eps 1e-8 \
--accelerator gpu --strategy deepspeed_stage_1 --grad_cp 0 --op $OP \
--precision bf16 \
--my_testing "x060" \
--train_type "state"  --dataload pad \
#--conv --conv_kernel 5  --conv_dummy_scale 0.00 \
# --lora --emb  \

# --cache_dir '/home/RWKV-ASR/src/data_cache/train'

# --quant $QUANT
