import torch

#模型超参数
dropout=0.1
block_size=256
theta=10000.0
eps=1e-5
layer_num=6
hidden_size=384
head_num=6
vocab_size=10000
use_flash_attention=True
ffn_hidden_size_multiplier=4
betas=(0.9, 0.95)
device="cuda" if torch.cuda.is_available() else "cpu"

#训练超参数
tokenizer_name="georgeyw/TinyStories-tokenizer-10k"
dataset_name="roneneldan/TinyStories"
batch_size=64
learning_rate=5e-4
max_iters=20000
eval_iters=200
weight_decay=1e-1
warmup_iters=1000
gradient_accumulation_steps=4
eval_interval=500

#推理超参数
temperature=0.8
max_token_size=256
