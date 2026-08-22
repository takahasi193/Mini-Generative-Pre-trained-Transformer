import torch
import torch.nn as nn
import torch.nn.functional as F
import config
import inspect



# 旋转位置编码
class RoPE(nn.Module):
    def __init__(self,theta,d_k,max_seq_len):
        super().__init__()
        # shape:[max_seq_len,d_k/2]
        self.freqs=torch.outer(torch.arange(max_seq_len).float(),1.0/theta**(torch.arange(0,d_k,2).float()/d_k))
        self.register_buffer("cos_table",torch.cos(self.freqs),persistent=False)
        self.register_buffer("sin_table",torch.sin(self.freqs),persistent=False)

    def forward(self,x:torch.Tensor,token_position:torch.Tensor)->torch.Tensor:
        # shape:[T,d_k/2]
        cos_t=self.cos_table[token_position].to(x.dtype)
        sin_t=self.sin_table[token_position].to(x.dtype)
        B,nh,T,d=x.shape
        x=x.reshape(B,nh,T,d//2,2)
        # shape:[B,nh,T,d_k/2]
        x0,x1=x[...,0],x[...,1]
        y=torch.stack([x0*cos_t-x1*sin_t,x0*sin_t+x1*cos_t],dim=-1)
        return y.reshape(B,nh,T,d)

class RMSnorm(nn.Module):
    def __init__(self,hidden_size,eps=config.eps):
        super().__init__()
        self.gamma=nn.Parameter(torch.ones(hidden_size))
        self.eps=eps
    def forward(self,x):
        input_type=x.dtype
        x=x.to(torch.float32)
        # shape:[B,T,1]
        rms=torch.rsqrt(x.pow(2).mean(dim=-1,keepdim=True)+self.eps)
        y=x*rms
        y=y*self.gamma
        return y.to(input_type)


class SwiGLU(nn.Module):
    def __init__(self,input_size,hidden_size):
        super().__init__()
        self.weight1=nn.Linear(input_size,hidden_size)
        self.weight2=nn.Linear(hidden_size,input_size)
        self.weight3=nn.Linear(input_size,hidden_size)
        self.silu=nn.SiLU()
    def forward(self,x):
        return self.weight2(self.silu(self.weight1(x))*self.weight3(x))
        


# 因果注意力
class CausalAttention(nn.Module):
    def __init__(self,hidden_size:int,head_num:int,block_size:int,use_flash_attention:bool=True):
        super().__init__()
        self.use_flash_attention=use_flash_attention
        self.attn=nn.Linear(hidden_size,3*hidden_size)
        self.head_num=head_num
        self.hidden_size=hidden_size
        self.dropout=nn.Dropout(config.dropout)
        self.proj=nn.Linear(hidden_size,hidden_size)
        self.rope=RoPE(config.theta,hidden_size//head_num,block_size)
        self.register_buffer("mask",torch.tril(torch.ones(block_size,block_size).view(1,1,block_size,block_size)))

    def forward(self,x:torch.Tensor)->torch.Tensor:
        B,T,C=x.shape 
        attn=self.attn(x)

        # 将attn切成三份分别表示q,k,v
        q,k,v=attn.split(self.hidden_size,dim=-1) 

        # 将q，k，v分头，同时将T和nh调换方便后面的运算 shape:[B,nh,T,d]
        q=q.view(B,T,self.head_num,C//self.head_num).transpose(1,2)
        k=k.view(B,T,self.head_num,C//self.head_num).transpose(1,2)
        v=v.view(B,T,self.head_num,C//self.head_num).transpose(1,2)

        # RoPE
        token_position=torch.arange(T,device=x.device)
        q=self.rope(q,token_position)
        k=self.rope(k,token_position)

        # 使用flash attention
        if self.use_flash_attention:
            y=F.scaled_dot_product_attention(q,k,v,attn_mask=None,dropout_p=config.dropout if self.training else 0,is_causal=True)

        # 不使用flash attention
        else:
            # shape:[B,nh,T,T]
            attn_map=q@k.transpose(2,3)
            scale=k.size(-1)**-0.5
            attn_map=attn_map*scale
            attn_map=attn_map.masked_fill(self.mask[...,:T,:T]==0,float("-inf"))
            attn_map=F.softmax(attn_map,dim=-1)
            attn_map=self.dropout(attn_map)
            # shape:[B,nh,T,d]
            y=attn_map@v

        y=y.transpose(1,2).contiguous().view(B,T,C)
        return self.proj(y)

# 前馈网络
class MLP(nn.Module):
    def __init__(self,hidden_size):
        super().__init__()
        self.ffn=nn.Sequential(
            SwiGLU(hidden_size,int(hidden_size*config.ffn_hidden_size_multiplier)),
            nn.Dropout(config.dropout)
        )


    def forward(self,x:torch.Tensor)->torch.Tensor:
        return self.ffn(x)

class TransformerBlock(nn.Module):
    def __init__(self,hidden_size,head_num,block_size):
        super().__init__()
        self.attn=CausalAttention(hidden_size,head_num,block_size)
        self.ffn=MLP(hidden_size)
        self.ln1=RMSnorm(hidden_size)
        self.ln2=RMSnorm(hidden_size)
        self.dropout=nn.Dropout(config.dropout)

    def forward(self,x:torch.Tensor)->torch.Tensor:
        x=x+self.dropout(self.attn(self.ln1(x)))
        y=x+self.dropout(self.ffn(self.ln2(x)))
        return y

# 模型的主体
class MiniGPT(nn.Module):
    def __init__(self,hidden_size,head_num,vocab_size,block_size,layer_num):
        super().__init__()
        self.block_size=block_size
        self.emd=nn.Embedding(vocab_size,hidden_size)
        self.blocks=nn.Sequential(*[TransformerBlock(hidden_size,head_num,block_size) for _ in range(layer_num)])
        self.ln_f=RMSnorm(hidden_size)
        self.dropout=nn.Dropout(config.dropout)
        self.proj=nn.Linear(hidden_size,vocab_size)
        self.proj.weight=self.emd.weight

    def forward(self,x:torch.Tensor,label:torch.Tensor =None)->torch.Tensor:
        B,T=x.shape
        # [B,T,hidden_size]
        x=self.emd(x)
        x=self.blocks(x)
        x=self.dropout(self.ln_f(x))
        logits=self.proj(x)

        if label is None:
          loss=None
        else:
          B,T,C=logits.shape
          shifted_logits = logits[...,:-1,:].reshape(B * (T-1), C)
          shifted_label = label[:,1:].reshape(B * (T-1))
          loss = F.cross_entropy(shifted_logits, shifted_label, ignore_index=-100)
        return logits,loss
            

    @torch.no_grad()
    def generate(self,idx:torch.Tensor,max_token_size,temperature=config.temperature):
        for _ in range(max_token_size):
            idx_cond=idx[...,-self.block_size:]
            logits,_=self(idx_cond)
            logits=logits[:,-1,:]/temperature
            prob=F.softmax(logits,dim=-1)
            new_token=torch.multinomial(prob,num_samples=1)
            idx=torch.cat((idx,new_token),dim=1)
        return idx


    def get_total_param_num(self):
        return sum([p.numel() for p in self.parameters()])

    def configure_optimizer(self,weight_decay,learning_rate,betas):
        param_dict={pn:p for pn,p in self.named_parameters() if p.requires_grad}
        decay_params=[p for _,p in param_dict.items() if p.dim()>=2]
        nondecay_params=[p for _,p in param_dict.items() if p.dim()<2]
        optim_groups=[
            {"params":decay_params,"weight_decay":weight_decay},
            {"params":nondecay_params,"weight_decay":0.0}
        ]
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and config.device == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        return optimizer


