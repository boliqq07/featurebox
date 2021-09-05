import torch
from torch import nn
from torch.nn import Parameter
nn.Linear
class MulLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super(MulLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(1, in_features, in_features, out_features, ))
        # if bias:
        #     self.bias = Parameter(torch.Tensor(out_features))
        # else:
        #     self.register_parameter('bias', None)
        # self.reset_parameters()

    def forward(self,x):
        x1 = torch.unsqueeze(x,dim=-1)
        x2 = torch.unsqueeze(x,dim=-2)
        temp = torch.matmul(x1,x2)
        temp = temp.unsqueeze(dim=-1)
        temp = self.weight * temp
        temp = torch.sum(temp,dim=1,keepdim=False)
        temp = torch.sum(temp,dim=1,keepdim=False)
        temp = torch.sum(temp,dim=-1,keepdim=False)
        temp = torch.sum(temp,dim=-1)
        return temp



x1 = torch.rand(size=(3,5),requires_grad=True)
mu = MulLayer(5,7)
s = mu(x1)
s.backward()
