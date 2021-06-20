import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, GRU

from torch_geometric.nn import NNConv, Set2Set


class CGGRUNet(torch.nn.Module):
    def __init__(self, num_node_features, num_edge_features, hidden_channels=128, hidden_channels2=64, n_conv=2,
                 n_set2set=2):
        super(CGGRUNet, self).__init__()
        dim = hidden_channels2
        self.lin0 = torch.nn.Linear(num_node_features, dim)

        nn = Sequential(Linear(num_edge_features, hidden_channels), ReLU(), Linear(hidden_channels, dim * dim))
        self.conv = NNConv(dim, dim, nn, aggr='mean')
        self.gru = GRU(dim, dim)

        self.set2set = Set2Set(dim, processing_steps=n_set2set) #very import
        self.lin1 = torch.nn.Linear(2 * dim, dim)
        self.lin2 = torch.nn.Linear(dim, 1)
        self.n_conv = n_conv

    def forward(self, data):
        out = F.relu(self.lin0(data.x))
        h = out.unsqueeze(0)

        for i in range(self.n_conv):
            m = self.conv(out, data.edge_index, data.edge_attr)
            m = F.relu(m)
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)

        out = self.set2set(out, data.batch)
        out = F.relu(self.lin1(out))
        out = self.lin2(out)
        return out.view(-1)
