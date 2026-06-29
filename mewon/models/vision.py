import torch
import torch.nn as nn

class ViT(nn.Module):
    def __init__(self,img=32,patch=4,classes=10,dim=128,depth=4,heads=4):
        super().__init__()
        self.patch=patch; n=(img//patch)**2
        self.patchify=nn.Conv2d(3,dim,kernel_size=patch,stride=patch,bias=False)
        self.pos=nn.Parameter(torch.zeros(1,n,dim))
        enc=nn.TransformerEncoderLayer(d_model=dim,nhead=heads,dim_feedforward=4*dim,activation='gelu',batch_first=True,norm_first=True)
        self.blocks=nn.TransformerEncoder(enc,depth)
        self.head=nn.Linear(dim,classes)
    def forward(self,x):
        x=self.patchify(x).flatten(2).transpose(1,2)+self.pos
        return self.head(self.blocks(x).mean(1))
