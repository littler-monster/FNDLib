import torch as th
from torch_scatter import scatter_mean
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import copy

class TDrumorGCN(th.nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super(TDrumorGCN, self).__init__()
        self.conv1 = GCNConv(in_feats, hid_feats)
        self.conv2 = GCNConv(hid_feats + in_feats, out_feats)

    def forward(self, data):
        device = data.x.device  # 动态获取当前设备
        x, edge_index = data.x, data.edge_index
        x1 = x.float().to(device)
        x = self.conv1(x, edge_index)
        x2 = x.clone().to(device)
        rootindex = data.rootindex.to(device)  # 确保 rootindex 在同一设备上
        root_extend = th.zeros(len(data.batch), x1.size(1)).to(device)  # 在设备上创建 root_extend
        batch_size = max(data.batch) + 1

        for num_batch in range(batch_size):
            index = th.eq(data.batch, num_batch)
            root_extend[index] = x1[rootindex[num_batch]]
        x = th.cat((x, root_extend), 1)

        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        root_extend = th.zeros(len(data.batch), x2.size(1)).to(device)
        for num_batch in range(batch_size):
            index = th.eq(data.batch, num_batch)
            root_extend[index] = x2[rootindex[num_batch]]
        x = th.cat((x, root_extend), 1)
        x = scatter_mean(x, data.batch, dim=0)

        return x


class BUrumorGCN(th.nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super(BUrumorGCN, self).__init__()
        self.conv1 = GCNConv(in_feats, hid_feats)
        self.conv2 = GCNConv(hid_feats + in_feats, out_feats)

    def forward(self, data):
        device = data.x.device  # 动态获取当前设备
        x, edge_index = data.x, data.BU_edge_index
        x1 = x.float().to(device)
        x = self.conv1(x, edge_index)
        x2 = x.clone().to(device)

        rootindex = data.rootindex.to(device)  # 确保 rootindex 在同一设备上
        root_extend = th.zeros(len(data.batch), x1.size(1)).to(device)  # 在设备上创建 root_extend
        batch_size = max(data.batch) + 1

        for num_batch in range(batch_size):
            index = th.eq(data.batch, num_batch)
            root_extend[index] = x1[rootindex[num_batch]]
        x = th.cat((x, root_extend), 1)

        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        root_extend = th.zeros(len(data.batch), x2.size(1)).to(device)
        for num_batch in range(batch_size):
            index = th.eq(data.batch, num_batch)
            root_extend[index] = x2[rootindex[num_batch]]
        x = th.cat((x, root_extend), 1)

        x = scatter_mean(x, data.batch, dim=0)
        return x


class BiGCN(th.nn.Module):
    def __init__(self, config):
        super(BiGCN, self).__init__()
        in_feats = config.graph_in_feats
        hid_feats = config.graph_hid_feats
        out_feats = config.graph_out_feats
        self.TDrumorGCN = TDrumorGCN(in_feats, hid_feats, out_feats)
        self.BUrumorGCN = BUrumorGCN(in_feats, hid_feats, out_feats)
        self.fc = th.nn.Linear((out_feats + hid_feats) * 2, 4)

    def forward(self, data):
        TD_x = self.TDrumorGCN(data)
        BU_x = self.BUrumorGCN(data)
        x = th.cat((BU_x, TD_x), 1)
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x
