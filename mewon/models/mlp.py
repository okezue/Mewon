import torch.nn as nn

class MLP(nn.Module):
    def __init__(self,indim=32,hidden=128,outdim=10,depth=3):
        super().__init__(); layers=[]; d=indim
        for _ in range(depth-1): layers+=[nn.Linear(d,hidden),nn.GELU()]; d=hidden
        layers.append(nn.Linear(d,outdim)); self.net=nn.Sequential(*layers)
    def forward(self,x): return self.net(x)
