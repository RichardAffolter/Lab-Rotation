# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# %%
'''
expected input: Tensor of dimensions N*C*in_features
output: Tensor of dimensions N*out_sets*⌈(in_features/m)⌉
N: batch size
C: number of channels (e.g. 4 for one-hot-encoding of SNPs)
in_features: number of input features (e.g. SNP positions)
out_sets: number of output sets (new channels)
m: how many in_features to group together
kernel_size: kernel of flat tensor: m*C
padding: should we padd at the end of the dataframe if in_features%m !=0? 
'''
class LocallyConnectedLayer(torch.nn.Module):
    def __init__(self, in_features, m, C=4, padding=True, bias=False, out_sets=4):
        super().__init__()
        self.in_features = in_features
        self.C = C
        self.m = m
        self.kernel_size = m*C
        self.padding = (m-(in_features%m))%m*C if padding else 0
        self.weight = nn.Parameter(torch.randn(1,self.kernel_size, out_sets))
        self.bias = nn.Parameter(torch.randn(1,out_sets)) if bias else None # with batchnorm we do not need bias
    
    def forward(self, x):
        x = x.float()
        x = x.transpose(-1,-2) # we need to transpose first to ensure that the channel values of one in_feature are next to each other after flattening
        x = x.flatten(1) # dim(N,in_features*C)
        x = F.pad(x,(0,self.padding))
        x = x.unfold(-1, size=self.kernel_size, step=self.kernel_size)
        x = torch.matmul(x,self.weight)
        if self.bias is not None:
            x = x+self.bias
        x = x.transpose(-1,-2) # transpose back to have the more convenient dimension order
        return x

# %%
class LCBlock(nn.Module):
    def __init__(self, in_features, m, out_sets=4, p=0.0):
        super().__init__()
        self.bn = nn.BatchNorm1d(out_sets)
        self.silu = nn.SiLU()
        self.drop = nn.Dropout(p)
        self.LCL1 = LocallyConnectedLayer(in_features, m=m, padding=True, out_sets=out_sets)
        self.LCL2 = LocallyConnectedLayer(in_features=math.ceil(in_features/m),m=m, padding=True, out_sets=out_sets)
        self.identity_downsample = nn.Linear(in_features, out_features=math.ceil(in_features/m**2)) if m!=1 else None

    def forward(self, x):
        identity = x

        x = self.bn(x)
        x = self.silu(x)
        x = self.LCL1(x)
        x = self.bn(x)
        x = self.silu(x)
        x = self.drop(x)
        x = self.LCL2(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x = x+identity
        return x

# %%
'''
expected input: flat tensor of shape N*in_features
expected output: flat tensor of shape N*out_features
N: batch size
'''
class FCBlock(nn.Module):
    def __init__(self, in_features, out_features, p=0.5):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(in_features)
        self.bn2 = nn.BatchNorm1d(out_features)
        self.silu = nn.SiLU()
        self.drop = nn.Dropout(p)
        self.FCL1 = nn.Linear(in_features=in_features, out_features=out_features)
        self.FCL2 = nn.Linear(in_features=out_features, out_features=out_features)
        self.identity_downsample = nn.Linear(in_features, out_features=out_features) if in_features != out_features else None

    def forward(self, x):
        identity = x

        x = self.bn1(x)
        x = self.silu(x)
        x = self.FCL1(x)
        x = self.bn2(x)
        x = self.silu(x)
        x = self.drop(x)
        x = self.FCL2(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x = x+identity
        return x
 # %%
class GLN(nn.Module):
    def __init__(self, in_features, num_classes, num_residual_blocks=2, m1=2, m2=2, C=4, num_predictor_blocks=4):
        super().__init__()
        self.m1 = m1
        self.m2 = m2
        self.num_residual_blocks = num_residual_blocks
        self.num_predictor_blocks = num_residual_blocks
        self.LCL0 = LocallyConnectedLayer(in_features, m=m1)
        Output1 = math.ceil(in_features/m1)
        self.LCLayers = self.make_LCLayers(Output1)
        Output2 = math.ceil(Output1/(2*m2)**num_residual_blocks)*C # we flatten after the last block TO DO: IMPLEMENT ENVIRONMENT CONCATENATION
        self.FCLayers = self.make_predictorLayers(in_features=Output2)
        self.bn = nn.BatchNorm1d(256)
        self.silu = nn.SiLU()
        self.drop = nn.Dropout(p=0.5)
        self.Linear = nn.Linear(256,num_classes)
        

    def make_LCLayers(self, in_features):
        layers = []
        for block in range(self.num_residual_blocks):
            layers.append(LCBlock(in_features=in_features, m=self.m2))
            in_features = math.ceil(in_features/self.m2**2)
        return nn.Sequential(*layers)

    def make_predictorLayers(self, in_features):
        layers = []
        layers.append(FCBlock(in_features=in_features, out_features=256))
        for block in range(self.num_predictor_blocks):
            layers.append(FCBlock(in_features=256, out_features=256))
        return nn.Sequential(*layers)

    
    def forward(self,x):
        x = self.LCL0(x)
        x = self.LCLayers(x)
        x = x.flatten(1)
        x = self.FCLayers(x)
        x = self.bn(x)
        x = self.silu(x)
        x = self.drop(x)
        x = self.Linear(x)
        return x

# %%
