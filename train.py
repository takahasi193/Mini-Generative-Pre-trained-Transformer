import model
import torch.nn as nn
import config
import torch
import os
from transformers import AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader



tokenizer=AutoTokenizer.from_pretrained(config.tokenizer_name)
datasets=load_dataset(config.dataset_name)

def tokenize_function(examples):
    return tokenizer(
        examples['text'], 
        truncation=True, 
        padding='max_length', 
        max_length=config.block_size
    )


tokenized_datasets = datasets.map(
    tokenize_function,
    batched=True,
    remove_columns=datasets['train'].column_names,
)
train_data=tokenized_datasets['train']
eval_data=tokenized_datasets['validation']
train_dataloader=DataLoader(train_data,batch_size=config.batch_size,shuffle=True)
eval_dataloader=DataLoader(eval_data,batch_size=config.batch_size,shuffle=False)

@torch.no_grad()
def estimate_loss(model:nn.Module):
    model.eval()
    out={}
    for split in ['train','eval']:
        loader=train_dataloader if split=='train' else eval_dataloader
        losses=torch.zeros(config.eval_iters)
        data_iter=iter(loader)
        for k in range(config.eval_iters):
             try:
              batch = next(data_iter)
             except StopIteration:
                data_iter = iter(loader)
                batch = next(data_iter)
             input_ids = batch['input_ids'].to(config.device)
             labels=input_ids.masked_fill(batch['attention_mask'].to(config.device) == 0, -100)
             _,loss=model(input_ids,labels)
             losses[k]=loss
        out[split]=losses.mean()
    model.train()
    return out 


# 定义模型
m=model.MiniGPT(config.hidden_size,config.head_num,config.vocab_size,config.block_size,config.layer_num).to(config.device)

# 定义优化器
optimizer=m.configure_optimizer(config.weight_decay,config.learning_rate,config.betas)

#
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.max_iters)

#保存路径
SAVE_PATH='./checkpoint'
os.makedirs(SAVE_PATH, exist_ok=True)

#最佳权重
best_eval_loss=float('inf')
best_ckpt=0

data_iter=iter(train_dataloader)
# 训练循环
for step in range(1,config.max_iters+1):
    if step % config.eval_interval == 0:
        losses = estimate_loss(m)
        print(f"step: {step}, train loss: {losses['train']:.4f}, eval loss: {losses['eval']:.4f}")
        checkpoint_file = os.path.join(SAVE_PATH, f"ckpt_step_{step}.pt")
        torch.save({
            "step": step,
            "model_state_dict": m.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "train_loss": losses["train"],
            "val_loss": losses["eval"],
        }, checkpoint_file)
        if losses["eval"] < best_eval_loss:
            best_eval_loss = losses["eval"]
            best_ckpt = step

    optimizer.zero_grad(set_to_none=True)
    for micro_step in range(config.gradient_accumulation_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_dataloader)
            batch = next(data_iter)

        input_ids = batch["input_ids"].to(config.device)
        label = input_ids.masked_fill(
            batch["attention_mask"].to(config.device) == 0,
            -100,
        )
        logits, loss = m(input_ids, label)
        (loss / config.gradient_accumulation_steps).backward()
    optimizer.step()
    scheduler.step()
   
    