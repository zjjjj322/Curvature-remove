import networkx as nx
from GraphRicciCurvature.OllivierRicci import OllivierRicci


class RicciCurvatureEdgeRemovalModel:
    def __init__(self, alpha=0.5, verbose="INFO"):
        """
        初始化 Ricci 曲率边删除模型。

        :param alpha: Ricci 曲率的计算参数，通常在 0 到 1 之间。
        :param verbose: 输出信息的详细程度。
        """
        self.alpha = alpha
        self.verbose = verbose

    def fit(self, G):
        """
        计算给定图的 Ricci 曲率，并删除曲率为负的边。

        :param G: 输入的 NetworkX 图对象。
        :return: 删除负曲率边后的新图。
        """
        # 初始化 OllivierRicci 对象
        orc = OllivierRicci(G, alpha=self.alpha, verbose=self.verbose)

        # 计算边的 Ricci 曲率
        curvature_edges = orc.compute_ricci_curvature_edges()

        # 筛选出曲率为负的边
        negative_curvature_edges = [edge for edge, curvature in curvature_edges.items() if curvature < 0]

        # 从图中删除这些边
        G.remove_edges_from(negative_curvature_edges)

        # 返回删除负曲率边后的图
        return G, curvature_edges, negative_curvature_edges

    def print_curvature_info(self, curvature_edges, negative_curvature_edges):
        """
        打印边的 Ricci 曲率信息以及被删除的边。

        :param curvature_edges: 所有边的曲率信息。
        :param negative_curvature_edges: 被删除的负曲率边列表。
        """
        print("负曲率边:")
        print(negative_curvature_edges)
        print("\n所有边的曲率:")
        for edge, curvature in curvature_edges.items():
            print(f"边 {edge} 的曲率: {curvature}")


# 使用示例
if __name__ == "__main__":
    # 创建空手道俱乐部图
    G = nx.karate_club_graph()

    # 初始化 Ricci 曲率模型
    model = RicciCurvatureEdgeRemovalModel(alpha=0.5, verbose="INFO")

    # 计算曲率并删除负曲率边
    G1, curvature_edges, negative_curvature_edges = model.fit(G)

    # 打印曲率信息
    model.print_curvature_info(curvature_edges, negative_curvature_edges)









from collections import defaultdict, namedtuple
from typing import Optional, Callable, Union

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import networkx as nx
from torch_scatter import scatter
from torch.nn import Parameter, Linear
from torch_geometric.nn import GCNConv, MessagePassing
from GraphRicciCurvature.OllivierRicci import OllivierRicci
from torch_geometric.utils import from_networkx, add_remaining_self_loops, softmax
from torch_sparse import coalesce



class RicciCurvaturePooling(nn.Module):
    def __init__(self, in_channels, alpha=0.5):
        super(RicciCurvaturePooling, self).__init__()
        self.alpha = alpha
        self.verbose = "ERROR"

    def forward(self, x, edge_index):
        # 将 edge_index 转换为 NetworkX 图
        G1 = self.convert_edge_index_to_graph(edge_index)

        # 计算边的 Ricci 曲率
        orc = OllivierRicci(G1, alpha=self.alpha, verbose=self.verbose)
        curvature_edges = orc.compute_ricci_curvature_edges()

        # 将无向图的边曲率双向存储
        full_curvature_edges = {}
        for (u, v), curvature in curvature_edges.items():
            full_curvature_edges[(u, v)] = curvature
            full_curvature_edges[(v, u)] = curvature

        # 筛选出曲率为负的边
        negative_edges = [edge for edge, curvature in full_curvature_edges.items() if curvature < 0]
        negative_edges_set = set(negative_edges)

        # 生成新的 edge_index，去除负曲率的边
        new_edge_index = []
        for (u, v) in G1.edges():
            if (u, v) not in negative_edges_set and (v, u) not in negative_edges_set:
                new_edge_index.append([u, v])
                new_edge_index.append([v, u])

        new_edge_index = torch.tensor(new_edge_index, dtype=torch.long).t().contiguous()
        return x, new_edge_index

    @staticmethod
    def convert_edge_index_to_graph(edge_index):
        """将 edge_index 转换为 NetworkX 图."""
        G0 = nx.Graph()
        edge_list = edge_index.t().tolist()
        G0.add_edges_from(edge_list)
        return G0


class GraphNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphNet, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.pool = RicciCurvaturePooling(alpha=0.5, in_channels=out_channels)

    def forward(self, G, x):
        data = from_networkx(G)

        edge_index = data.edge_index
        print(edge_index, '\n', edge_index.shape)
        # 第一层卷积
        x = F.relu(self.conv1(x, edge_index))

        # 应用池化层，删除曲率为负的边，更新后的 edge_index 和 x 会传递给下一层
        x, edge_index = self.pool(x, edge_index)

        return x, edge_index

#G = nx.karate_club_graph()
G = nx.barabasi_albert_graph(50, 3)
num_nodes = G.number_of_nodes()

# 生成独热编码的特征矩阵
feature = torch.eye(num_nodes, dtype=torch.float)

# 创建模型并运行前向传播
model = GraphNet(in_channels=num_nodes, hidden_channels=64, out_channels=1)
out, updated_edge_index = model(G, feature)

print(out.shape)  # 输出的特征矩阵形状
print(out)  # 输出的特征矩阵
print(updated_edge_index)  # 输出更新后的边索引
#print(updated_edge_index.shape)
#print(unpool_info1)  # 输出池化信息
#print(fitness1)  # 输出 fitness 信息

