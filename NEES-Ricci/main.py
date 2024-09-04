import copy
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import torch.optim as optim
from torch_geometric.nn import GCNConv
from torch_geometric.utils import from_networkx

from Graphtools import generate_and_save_ba_network, createGraphFromGML, normalize_adj, sparse_mx_to_torch_sparse_tensor
from Pooling1 import RicciCurvaturePooling1
from Pooling3 import RicciCurvaturePooling3
import scipy.sparse as sp

class GraphNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphNet, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, out_channels)
        self.pool = RicciCurvaturePooling3(alpha=0.5, in_channels=out_channels)
        #self.pool = RicciCurvaturePooling1(alpha=0.5, in_channels=out_channels)
        self.conv4 = GCNConv(out_channels, 1)

    def forward(self, G, x):
        data = from_networkx(G)

        edge_index = data.edge_index
        #print(edge_index, '\n', edge_index.shape)
        print(edge_index.shape)
        # 第一层卷积
        x = F.relu(self.conv1(x, edge_index))
        # 第二层卷积
        x = F.relu(self.conv2(x, edge_index))
        # 第三层卷积
        x = F.relu(self.conv3(x, edge_index))
        # 应用池化层，删除曲率为负的边，更新后的 edge_index 和 x 会传递给下一层
        x_pool, new_edge_index1, unpool_info, fitness, loss1 = self.pool(x, edge_index)
        #x, new_edge_index1, unpool_info, cluster, fitness, loss = self.pool(x, edge_index)
        x1 = F.relu(self.conv4(x_pool, new_edge_index1))
        #加一个sigmoid函数
        x1 = torch.sigmoid(x1)

        #return x1, new_edge_index1, unpool_info, cluster, fitness, loss
        return x1, new_edge_index1, unpool_info, fitness, loss1


class GraphNet1(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphNet1, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, out_channels)
        self.pool = RicciCurvaturePooling1(alpha=0.5, in_channels=out_channels)
        self.conv4 = GCNConv(out_channels, 1)

    def forward(self, G, x):
        data = from_networkx(G)

        edge_index = data.edge_index
        #print(edge_index, '\n', edge_index.shape)
        print(edge_index.shape)
        # 第一层卷积
        x = F.relu(self.conv1(x, edge_index))
        # 第二层卷积
        x = F.relu(self.conv2(x, edge_index))
        # 第三层卷积
        x = F.relu(self.conv3(x, edge_index))
        # 应用池化层，删除曲率为负的边，更新后的 edge_index 和 x 会传递给下一层
        x_pool, new_edge_index1, unpool_info, fitness, loss1 = self.pool(x, edge_index)
        #x, new_edge_index1, unpool_info, cluster, fitness, loss = self.pool(x, edge_index)
        x1 = F.relu(self.conv4(x_pool, new_edge_index1))
        #加一个sigmoid函数
        x1 = torch.sigmoid(x1)

        #return x1, new_edge_index1, unpool_info, cluster, fitness, loss
        return x1, new_edge_index1, unpool_info, fitness, loss1




def main():

    gml_dir = "data/"
    FILE_NET = ["football.txt"]

    ep = 210

    x = {}
    y = {}
    z = {}

    # 创建存储GML文件的目录
    if not os.path.exists(gml_dir):
        os.makedirs(gml_dir)

    for f in range(len(FILE_NET)):

        # 生成和保存BA网络（例如，n=100，m=3）
        gml_file_path = os.path.join(gml_dir, f"ba_network_{f}.gml")
        generate_and_save_ba_network(gml_file_path, n=50, m=3)

        # 加载GML文件并创建图
        G = createGraphFromGML(gml_file_path)
        A = nx.adjacency_matrix(G).todense()
        num_nodes = G.number_of_nodes()

        norm_adj = normalize_adj(A)  # Normalization using D^(-1/2) A D^(-1/2)
        adj = sparse_mx_to_torch_sparse_tensor(sp.csr_matrix(norm_adj)).to_dense()

        feature = torch.eye(num_nodes, dtype=torch.float)

        x[f] = []
        y[f] = []
        z[f] = []
        #min_loss = float('inf')
        best_epoch = 0

        model = GraphNet(in_channels=num_nodes, hidden_channels=64, out_channels=32)
        optimizer = optim.Adam(model.parameters(), lr=0.0005)

        for epoch in range(ep):
            t = time.time()
            model.train()
            G0 = copy.deepcopy(G)

            # 测试模型
            loss = model(G0, feature)[4]

            x[f].append(epoch)
            z[f].append(loss.item())

            # 检查是否为当前最小的loss
            if epoch > 50 and loss.item() < min(z[f]):
                min_loss = loss
                best_epoch = epoch
                print("save")
                torch.save(model.state_dict(), f"./result-trial-{FILE_NET[f][:-4]}.pt")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print("epoch:", '%04d' % (epoch + 1), "train_loss=",
                  "{:.5f}".format(loss), "time=", "{:.5f}".format(time.time() - t))

        print(f"训练结束，最小loss出现在第 {best_epoch + 1} 个epoch，loss值为 {min_loss:.5f}")

        model.load_state_dict(torch.load(f"./result-trial-{FILE_NET[f][:-4]}.pt"))
        loss = model(G, feature, adj, model)[4]
        print("测试完成，最终损失值为:", loss)


if __name__ == "__main__":
    main()
