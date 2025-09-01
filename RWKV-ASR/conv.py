import torch 
import numpy as np
import torch.nn as nn 

x= [[[1,2,3],[4,5,6],[1,1,2],[1,1,3]]]
x = torch.tensor(x, dtype=torch.float32)
print(x.shape)
x = x.permute(0, 2, 1)
conv=nn.Conv1d(3,2,2)
x = conv(x)
x = x.permute(0, 2, 1)
print(x)