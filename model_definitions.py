import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_max_pool
import torchvision.models as models




# ✅ Define Model Architectures

class CNNBranch(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.feature_extractor = nn.Sequential(*list(base.children())[:-1])  # remove last FC layer
        self.fc = nn.Linear(512, 600)

    def forward(self, x):
        x = self.feature_extractor(x).view(x.size(0), -1)
        return self.fc(x)

class HierarchicalGNNBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.fds = [80, 160, 400, 600]
        self.mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(3072 if i == 0 else self.fds[i - 1], fd * 4),
                nn.ReLU(),
                nn.Dropout(0.6),
                nn.Linear(fd * 4, fd),
                nn.ReLU(),
                nn.Dropout(0.6)
            ) for i, fd in enumerate(self.fds)
        ])
        self.convs = nn.ModuleList([GCNConv(fd, fd) for fd in self.fds])
        self.norms = nn.ModuleList([nn.BatchNorm1d(fd) for fd in self.fds])
        self.dropout = nn.Dropout(0.6)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for i in range(len(self.fds)):
            x_res = x
            x = self.mlps[i](x)
            x = self.convs[i](x, edge_index)
            x = self.norms[i](x)
            x = F.relu(x + x_res) if x.shape == x_res.shape else F.relu(x)
            x = self.dropout(x)
        return global_max_pool(x, batch)

class FuNetA(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = CNNBranch()
        self.gnn = HierarchicalGNNBranch()
        self.fc = nn.Linear(600, 2)

    def forward(self, img, graph):
        return self.fc(self.cnn(img) + self.gnn(graph))

class FuNetM(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = CNNBranch()
        self.gnn = HierarchicalGNNBranch()
        self.fc = nn.Linear(600, 2)

    def forward(self, img, graph):
        return self.fc(self.cnn(img) * self.gnn(graph))

class FuNetC(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = CNNBranch()
        self.gnn = HierarchicalGNNBranch()
        self.fc = nn.Linear(1200, 2)

    def forward(self, img, graph):
        return self.fc(torch.cat([self.cnn(img), self.gnn(graph)], dim=1))