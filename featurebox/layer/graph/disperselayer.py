from cmath import pi

import torch
from torch import nn, Tensor


from bgnet.layer.graph.baselayer import BaseLayer
from bgnet.preprocessing.core.kpath import HighSymPointsCPP


class DisperseDataLayer(nn.Module):

    def __init__(self, out_channels: int = 32):
        super().__init__()
        self.out_channels = out_channels

        self.conv = nn.Conv2d(in_channels=1,
                              out_channels=out_channels,
                              kernel_size=3,
                              stride=1,
                              padding=0)
        self.drop = nn.Dropout2d(0.2)
        self.bt2 = nn.BatchNorm2d(out_channels)
        self.convert = HighSymPointsCPP()

    def disperse(self, sgt:Tensor):
        device = sgt.device
        sgt = sgt.to("cpu")
        sgt = self.convert.kps(sgt)
        sgt = sgt.to(device)
        return torch.cos(sgt*2*pi)

    def forward(self, oe_interacted, sgt, wei):
        """

        Args:
            oe_interacted: torch.tensor shape(N_node,20,3)
            sgt:torch.tensor shape
            wei: torch.tensor shape

        Returns:

        """
        sgt = self.disperse(sgt)
        wei = torch.unsqueeze(wei,1)
        wei = torch.repeat_interleave(wei,20,1)
        sgt = oe_interacted * sgt * wei
        sgt = torch.sum(sgt, dim=-1, keepdim=False)

        return sgt


class InitAtomEnergy(BaseLayer):
    def __init__(self,inner_channels=20):
        super().__init__()
        self.linear1 = nn.Linear(4, inner_channels)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(inner_channels,1)

    def forward(self, ie_fea, node_atom_idx):
        """

        Args:
            ie_fea: torch.tensor shape(N,ie_fea_len)
            node_atom_idx: list of tensor

        Returns:
            ie_fea: torch.tensor shape(N_node,1)
        """
        ie_fea = torch.cat([
            self.merge_idx(ie_fea, node_atom_idx, method ="mean"),
            self.merge_idx(ie_fea, node_atom_idx, method ="sum"),
            self.merge_idx(ie_fea, node_atom_idx, method = "max"),
            self.merge_idx(ie_fea, node_atom_idx, method = "min"),
         ], dim=1)
        x = self.linear1(ie_fea)
        x = self.relu(x)
        x = self.linear2(x)

        return x
