import numpy as np
import torch
from torch import Tensor, nn, tensor

import torch.nn.functional as F
from bgnet.layer.graph.baselayer import BaseLayer

sqrtpi = np.sqrt(2 * np.pi)


def pdf(x, mu, sigma):
    return torch.exp(-(x - mu) ** 2 / (2.0 * sigma ** 2) )/ (sqrtpi * sigma)


def pdf_all(x, mu, sigma):
    return [torch.exp(-(x - mui) ** 2/ (2.0 * sigma ** 2)) / (sqrtpi * sigma) for mui in mu]


class _DistributionLayer(BaseLayer):
    """For change the value to Gauss Distribution"""
    def __init__(self, miny, maxy, sums=True, disperse_number=10, sd=0.3):
        super(_DistributionLayer, self).__init__()
        if sums:
            self.func = self.discre_sum
        else:
            self.func = self.discre
        self.miny = miny
        self.maxy = maxy
        self.disperse_number = disperse_number
        self.sd = sd

    @staticmethod
    def discre(y, miny, maxy, disperse_number=10, sd=0.3):

        device = y.device
        y = y.to("cpu")
        y = [yi for yi in y if 0 < yi < 110]

        step = torch.linspace(miny, maxy, steps=disperse_number)
        new_y = pdf_all(step, y, sd)

        new_y = torch.stack([yi / torch.max(yi) if torch.max(yi) != 0 else yi for yi in new_y])

        new_y = new_y.to(dtype=torch.float32, device=device)
        return new_y

    def discre_sum(self, y, miny, maxy, disperse_number=10, sd=0.3):
        new_y = self.discre(y, miny, maxy, disperse_number, sd)
        sum_x = torch.sum(new_y, dim=0, keepdim=True)
        return sum_x/torch.max(sum_x)

    def forward(self, x):
        x = self.func(x, self.miny, self.maxy, self.disperse_number, self.sd, )
        return x


class DistributionDataLayer(BaseLayer):
    """For change the value to Gauss Distribution hear map"""
    def __init__(self, miny, maxy, disperse_number=10, sd=0.3):
        super().__init__()

        self.miny = miny
        self.maxy = maxy
        self.disperse_number = disperse_number
        self.sd = sd

        self.distribute = _DistributionLayer(miny, maxy, sums=True, disperse_number=disperse_number, sd=sd)

    def forward(self, oe, node_ele_idx, ele_atom_idx):
        """
        Args:
            oe: (torch.Tensor) torch.float32, shape (N, 17)
            node_ele_idx: (list of torch.Tensor) torch.int64, each one shape is different.
            ele_atom_idx: (list of torch.Tensor) torch.int64, each one shape is different.

        Returns:
            torch.tensor shape(N_node,1,disperse_number,disperse_number)
        """
        oe = self.merge_idx(oe, node_ele_idx)
        oe = oe/torch.linalg.norm(oe, dim=1).view(-1,1)
        oe = self.merge_idx(oe, ele_atom_idx, method="sum")

        oe = torch.cat([self.distribute(i) for i in oe])

        a = torch.repeat_interleave(torch.unsqueeze(oe, dim=1),20,1)
        b = torch.repeat_interleave(torch.unsqueeze(oe, dim=2),20,2)
        oe = a+b

        oe = torch.unsqueeze(oe, 1)
        return oe


class InteractLayer(nn.Module):

    def __init__(self, out_channels: int = 32):
        super().__init__()
        self.out_channels = out_channels
        self.dn = 20
        self.dis = DistributionDataLayer(0, 50, self.dn, sd=0.3)

        self.conv = nn.Conv2d(in_channels=1,
                              out_channels=out_channels,
                              kernel_size=3,
                              stride=1,
                              padding=1)
        self.drop = nn.Dropout2d(0.2)
        self.bt2 = nn.BatchNorm2d(out_channels)
        self.linear1 = nn.Linear(out_channels*self.dn*self.dn,500)
        self.linear2 = nn.Linear(500, 60)

    def forward(self, oe, node_ele_idx, ele_atom_idx):
        """
        Args:
            oe: (torch.Tensor) torch.float32, shape (N, 17)
            node_ele_idx: (list of torch.Tensor) torch.int64, each one shape is different.
            ele_atom_idx: (list of torch.Tensor) torch.int64, each one shape is different.

        Returns:
            torch.tensor shape(N_node,20,3)
        """
        x1 = self.dis(oe, node_ele_idx, ele_atom_idx)
        x = self.conv(x1)
        x = self.drop(x)
        x = self.bt2(x) + torch.repeat_interleave(x1, self.out_channels, 1)
        x = torch.flatten(x,1)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = torch.reshape(x,(x.shape[0],20,3))
        return x


if __name__ == "__main__":
    li = _DistributionLayer(0, 5,disperse_number=50,sd =0.2)
    ass = [torch.tensor([0.233]),
           torch.tensor([0.570]),
           torch.tensor([9.948,0.501,0.199]),
           torch.tensor([5.075,3.515,0.398,0.153]),
           torch.tensor([3.095,2.527,1.6,0.205,0.176]),
           ]
    a= np.concatenate([li(a).numpy() for a in ass])
    a = np.sum(a,0)
    b = a-a.reshape(-1,1)
    # device = torch.device('cuda:0')
#
# dis = DispersionLayer()
