import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Attn(nn.Module):
    def __init__(self,nembd,nhead,blk):
        super().__init__(); assert nembd%nhead==0
        self.nhead=nhead; self.hdim=nembd//nhead
        self.qkv=nn.Linear(nembd,3*nembd,bias=False); self.proj=nn.Linear(nembd,nembd,bias=False)
        self.register_buffer('mask',torch.tril(torch.ones(blk,blk,dtype=torch.bool)).view(1,1,blk,blk),persistent=False)
    def forward(self,x):
        B,T,C=x.shape; q,k,v=self.qkv(x).chunk(3,dim=-1)
        q=q.view(B,T,self.nhead,self.hdim).transpose(1,2); k=k.view(B,T,self.nhead,self.hdim).transpose(1,2); v=v.view(B,T,self.nhead,self.hdim).transpose(1,2)
        att=(q@k.transpose(-2,-1))/math.sqrt(self.hdim); att=att.masked_fill(~self.mask[:,:,:T,:T],float('-inf')).softmax(-1)
        return self.proj((att@v).transpose(1,2).contiguous().view(B,T,C))

class MLP(nn.Module):
    def __init__(self,nembd):
        super().__init__(); self.fc=nn.Linear(nembd,4*nembd,bias=False); self.proj=nn.Linear(4*nembd,nembd,bias=False)
    def forward(self,x): return self.proj(F.gelu(self.fc(x)))

class Block(nn.Module):
    def __init__(self,nembd,nhead,blk):
        super().__init__(); self.ln1=nn.LayerNorm(nembd); self.attn=Attn(nembd,nhead,blk); self.ln2=nn.LayerNorm(nembd); self.mlp=MLP(nembd)
    def forward(self,x): return x+self.mlp(self.ln2(x+self.attn(self.ln1(x))))

class GPT(nn.Module):
    def __init__(self,vocab,blk,nlayer=2,nhead=2,nembd=64,drop=0.0):
        super().__init__(); self.blk=blk
        self.wte=nn.Embedding(vocab,nembd); self.wpe=nn.Embedding(blk,nembd)
        self.blocks=nn.ModuleList([Block(nembd,nhead,blk) for _ in range(nlayer)])
        self.lnf=nn.LayerNorm(nembd); self.head=nn.Linear(nembd,vocab,bias=False)
    def forward(self,idx,targets=None):
        B,T=idx.shape
        if T>self.blk: raise ValueError('sequence too long')
        x=self.wte(idx)+self.wpe(torch.arange(T,device=idx.device))
        for b in self.blocks: x=b(x)
        logits=self.head(self.lnf(x)); loss=None
        if targets is not None: loss=F.cross_entropy(logits.reshape(B*T,-1),targets.reshape(B*T))
        return logits,loss
