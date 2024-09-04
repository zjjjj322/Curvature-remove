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

from Pooling1 import RicciCurvaturePooling1


class simiConv(MessagePassing):  # self.gnn_score = simiConv(self.in_channels, 1)
    def __init__(self, in_channels, out_channels):
        super(simiConv, self).__init__(aggr='add')  #'mean'

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.lin1 = Linear(in_channels, in_channels, bias=False)
        self.lin2 = Linear(in_channels, out_channels, bias=True)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x, edge_index):  #这里传入的x就是前面得到的h向量
        """"""
        a = self.lin1(x)  #对应公式7的W1，是一个线性变化即乘上W1（dxd维）
        b = self.lin2(x)  #对应公式8得W2（dx1维）参数，他这里都是先乘上参数再做运算
        out = self.propagate(edge_index, x=a, x_cluster=b)  #x_cluster就是h向量乘上W2
        # return out + b
        return out  #.sigmoid()

    def message(self, x_i, x_j, x_cluster_i):
        out = torch.cosine_similarity(x_i, x_j).reshape(-1, 1)  #获得节点相似性方法可以换，x_i表i节点本身，x_j表i节点得邻居
        print(x_i.shape, out.shape)
        return x_cluster_i * out  #这里就是公式8，x_cluster_i就是i节点的x_cluster，out是eij
        # return  out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


class RicciCurvaturePooling(nn.Module):
    unpool_description = namedtuple(
        "UnpoolDescription",
        ["edge_index", "cluster"])

    def __init__(self, in_channels: int, ratio: Union[float, int] = 0.01, alpha=0.5, GNN: Optional[Callable] = GCNConv,
                 dropout: float = 0.0, negative_slope: float = 0.2, add_self_loops: bool = False, **kwargs):
        super(RicciCurvaturePooling, self).__init__()
        self.alpha = alpha
        self.in_channels = in_channels
        self.ratio = ratio
        self.verbose = "ERROR"
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.GNN = GNN
        self.add_self_loops = add_self_loops
        self.heads = 6
        self.weight = Parameter(torch.Tensor(in_channels, self.heads * in_channels))
        self.attention = Parameter(torch.Tensor(1, self.heads, 2 * in_channels))
        self.use_attention = True

        self.lin = Linear(in_channels, in_channels)
        self.att = Linear(2 * in_channels, in_channels)
        self.gnn_score = simiConv(self.in_channels, 1)
        self.marginloss = nn.MarginRankingLoss(0.5)
        self.BCEWloss = nn.BCEWithLogitsLoss()
        self.CosineLoss = nn.CosineEmbeddingLoss(margin=0.2)
        if self.GNN is not None:
            self.gnn_intra_cluster = GNN(self.in_channels, self.in_channels,
                                         **kwargs)
        self.reset_parameters()

    def glorot(self, tensor):  # inits.py中
        if tensor is not None:
            stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
            tensor.data.uniform_(-stdv, stdv)

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.att.reset_parameters()
        self.gnn_score.reset_parameters()
        self.glorot(self.weight)
        self.glorot(self.attention)
        if self.GNN is not None:
            self.gnn_intra_cluster.reset_parameters()

    def gen_subs(self, edge_index, N):  #找出节点的邻接节点并将其Conv(self.in_channels, 1)作为子图，并返回相应的索引
        edgelists = defaultdict(list)  # 用于存储每个节点的邻接节点
        match = defaultdict(list)
        for i in range(edge_index.size()[1]):
            s = int(edge_index[0][i])
            t = int(edge_index[1][i])
            if s != t:
                edgelists[s].append(t)

        start = []
        end = []
        for i in range(N):
            start.append(i)
            end.append(i)
            match[i].append(i)  #这里可以去掉就不含自身---------
            if len(match[i]) == 1:
                match[i].extend(edgelists[i])
                start.extend(edgelists[i])
                end.extend([i] * len(edgelists[i]))

        #start = []
        #end = []
        #for i in range(N):
        #start.append(i)
        #end.append(i)
        #if i in edgelists:
        #for j in edgelists[i]:
        #start.append(j)
        #end.append(i)

        source_nodes = torch.Tensor(start).reshape((1, -1))
        target_nodes = torch.Tensor(end).reshape((1, -1))
        subindex = torch.tensor(np.concatenate((source_nodes, target_nodes), axis=0), dtype=torch.long)

        return subindex, edgelists, match

    def choose(self, x, x_pool, edge_index, batch, score, match):
        nodes_remaining = set(range(x.size(0)))
        cluster = torch.empty_like(batch, device=torch.device('cpu'))
        node_argsort = torch.argsort(score, descending=True)
        i = 0
        transfer = {}
        new_node_indices = []
        tar_in = []
        tar_tar = []
        for node_idx in node_argsort.tolist():  #sort_node 这一次迭代表示将motif合并并用一个新的索引表示这个合并的节点，同时移除原有的这些节点防止重叠
            source = match[node_idx]

            d = [True for c in source if c not in nodes_remaining]
            if d:
                # print(1)
                continue

            transfer[i] = node_idx  # transfer表示融合后的

            new_node_indices.append(node_idx)
            for j in source:
                cluster[j] = i  #记录哪些节点j被融合到新节点序号i中
                if j != node_idx:
                    tar_in.append(j)
                    tar_tar.append(i)

            nodes_remaining = [j for j in nodes_remaining if j not in source]

            i += 1

        for node_idx in nodes_remaining:
            cluster[node_idx] = i
            transfer[i] = node_idx
            i += 1

        #cluster = cluster.to(torch.device('cuda'))
        cluster = cluster.to(x.device)
        index = new_node_indices + nodes_remaining  #融合后还有哪些节点
        new_x_pool = x_pool[index, :]
        new_x = torch.cat([x[new_node_indices, :], x_pool[nodes_remaining, :]])
        new_score = score[new_node_indices]
        if len(nodes_remaining) > 0:
            remaining_score = x.new_ones(
                (new_x.size(0) - len(new_node_indices),))
            new_score = torch.cat([new_score, remaining_score])
        new_x = new_x * new_score.view(-1, 1)
        N = new_x.size(0)
        new_edge_index, _ = coalesce(cluster[edge_index], None, N, N)  #用聚类中的序号替换原来的节点索引序号
        unpool_info = self.unpool_description(edge_index=edge_index,
                                              cluster=cluster)

        #生成正样本和负样本
        pos = []
        anchor_pos = []
        neg = []
        anchor_neg = []
        sig = {}
        for idx in range(x.size(0)):
            sig[idx] = []
            if cluster[idx].item() in range(len(transfer)):  # 生成正样本
                pos.append(idx)
                anchor_pos.append(cluster[idx].item())
                for other_cluster in range(len(transfer)):  # 遍历所有其他聚类
                    if other_cluster != cluster[idx].item() and other_cluster not in sig[idx]:
                        # 生成负样本：选择与当前节点不同簇的所有聚类
                        neg.append(idx)
                        anchor_neg.append(other_cluster)
                        sig[idx].append(other_cluster)

        pos_pos = x_pool[pos]
        pos_anchor = new_x[anchor_pos]
        neg_neg = x_pool[neg]
        neg_anchor = new_x[anchor_neg]

        return new_x, new_x_pool, new_edge_index, unpool_info, cluster, transfer, pos_pos, pos_anchor, neg_neg, neg_anchor

    def BCEloss(self, pos_anchor, pos, neg_anchor, neg):
        n1, h1 = pos_anchor.size()
        n2, h2 = neg_anchor.size()

        TotalLoss = 0.0
        pos = torch.bmm(pos_anchor.view(n1, 1, h1), pos.view(n1, h1, 1))
        loss1 = self.BCEWloss(pos, torch.ones_like(pos))
        if neg_anchor.size()[0] != 0:
            neg = torch.bmm(neg_anchor.view(n2, 1, h2), neg.view(n2, h2, 1))
            loss2 = self.BCEWloss(neg, torch.zeros_like(neg))
        else:
            loss2 = 0

        TotalLoss += loss2 + loss1
        return TotalLoss

    def forward(self, x, edge_index, batch=None):
        # 将 edge_index 转换为 NetworkX 图
        G = self.convert_edge_index_to_graph(edge_index)
        N = x.size(0)
        if N == 1:
            unpool_info = self.unpool_description(edge_index=edge_index,
                                                  cluster=torch.tensor([0]))
            return x, edge_index, unpool_info, torch.tensor(0.0, requires_grad=True), 0.0

        edge_index, _ = add_remaining_self_loops(edge_index, fill_value=1, num_nodes=N)

        if batch is None:
            batch = torch.LongTensor(size=([N]))

        subindex, edgelists, match = self.gen_subs(edge_index, N)
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        x_pool = x
        if self.GNN is not None:
            print("GNN")
            x_pool_j = self.gnn_intra_cluster(x=x, edge_index=edge_index)  # 这里用了自注意力gnn对x处理
        print("x_pool", x_pool.size())

        if self.use_attention:
            x_pool_j = torch.matmul(x_pool_j, self.weight)

            x_pool_j = x_pool_j.view(-1, self.heads, self.in_channels)

            x_i = x_pool_j[subindex[0]]  # 保存原来邻居的特征向量xj

            x_j = scatter(x_i, subindex[1], dim=0, reduce='max')  # 虚拟合并节点xmi向量

            alpha = (torch.cat([x_i, x_j[subindex[1]]], dim=-1) * self.attention).sum(dim=-1)  # 对应公式4向量拼接部分

            alpha = F.leaky_relu(alpha, self.negative_slope)  # self.negative_slope=0.2
            alpha = softmax(alpha, subindex[1], num_nodes=x_pool_j.size(0))
            alpha = F.dropout(alpha, p=self.dropout, training=self.training)

            v_j = x_pool_j[subindex[0]] * alpha.view(-1, self.heads, 1)

            x = scatter(v_j, subindex[1], dim=0, reduce='add')

            x = x.mean(dim=1)  # 就是h向量

        fitness = self.gnn_score(x, subindex).sigmoid().view(-1)
        x, new_x_pool, new_edge_index, unpool_info, cluster, transfer, pos_pos, pos_anchor, neg_neg, neg_anchor = self.choose(
            x, x_pool, edge_index, batch, fitness, match)

        loss = self.BCEloss(pos_anchor, pos_pos, neg_anchor, neg_neg)

        # 计算边的 Ricci 曲率
        orc = OllivierRicci(G, alpha=self.alpha, verbose=self.verbose)
        curvature_edges = orc.compute_ricci_curvature_edges()

        # 将无向图的边曲率双向存储
        full_curvature_edges = {}
        for (u, v), curvature in curvature_edges.items():
            full_curvature_edges[(u, v)] = curvature
            full_curvature_edges[(v, u)] = curvature

        set1 = {}
        for i in G.nodes():
            i1 = int(unpool_info.cluster[i])
            if i1 not in set1.keys():
                set1[i1] = []
            set1[i1].append(i)

        back = {}
        restored_edges = []
        # 构建 back 字典，记录融合后节点与原始节点的映射
        for i in range(len(cluster)):
            if int(cluster[i]) in back:
                back[int(cluster[i])].append(i)
            else:
                back[int(cluster[i])] = [i]
        # 遍历融合图中的每条边
        for idx in range(new_edge_index.size()[1]):
            s = int(new_edge_index[0][idx])  # 融合图中的源节点
            t = int(new_edge_index[1][idx])  # 融合图中的目标节点
            if s in list(set1.keys()) and t in list(set1.keys()) and s != t:
                flags = 0
                flagt = 0
                if len(back[s]) > 1:
                    flags = 1
                if len(back[t]) > 1:
                    flagt = 1
                if flags or flagt:
                    for i in set1[s]:  # 进行还原，每次根据聚类的信息和合并后的edge_index将motif还原成未合并的节点，并查看这些节点的邻居是否在该节点合并的motif相连接的motif中如果有就加上权重
                        for j in list(G.neighbors(i)):
                            if j in set1[t]:
                                restored_edges.append((i, j))
                                #restored_edges.append((j, i))#再加这个就添加了两遍了本来每个节点就会迭代一次再添加一遍就会重复两次--------------------------------
                                #判断restored_edges的边曲率是否为负，若为负删除这些边并从edge_index中删除这些边
        negative_edges = [edge for edge in restored_edges if full_curvature_edges[edge] < 0]
        negative_edges_set = set(negative_edges)  #这里是为了消除重复边但上面已经少加了所以可以删掉这个--------------------------------
        #打印移除边的数量和曲率
        #打印原图的边索引数量
        #print(2 * len(G.edges))  #这里的edge_index是哪个怎么每次运行都不变？--------------------------
        #print(len(negative_edges))
        #for edge in negative_edges:
        #   print(full_curvature_edges[edge])
        new_edge_index = []
        for (u, v) in G.edges():
            if (u, v) not in negative_edges_set and (
            v, u) not in negative_edges_set:  #debug检查new_edge_index是哪个要不要换名字写成new_edge_index1啥的
                new_edge_index.append([u, v])
                new_edge_index.append([v, u])

        # 将列表转换为 NumPy 数组
        new_edge_index = np.array(new_edge_index)

        # 转置数组，使其成为形状为 [2, num_edges] 的数组
        new_edge_index = new_edge_index.T

        # 将 NumPy 数组转换为 PyTorch 张量
        new_edge_index = torch.tensor(new_edge_index, dtype=torch.long)

        return x_pool, new_edge_index, unpool_info, fitness, loss
        #return x, new_edge_index, unpool_info, cluster, fitness, loss

    @staticmethod
    def convert_edge_index_to_graph(edge_index):
        #将 edge_index 转换为 NetworkX 图
        G = nx.Graph()
        edge_list = edge_index.t().tolist()
        G.add_edges_from(edge_list)
        return G


class GraphNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphNet, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, out_channels)
        self.pool = RicciCurvaturePooling(alpha=0.5, in_channels=out_channels)
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
        x_pool, new_edge_index1, unpool_info, fitness, loss = self.pool(x, edge_index)
        #x, new_edge_index1, unpool_info, cluster, fitness, loss = self.pool(x, edge_index)
        x1 = F.relu(self.conv4(x_pool, new_edge_index1))

        #return x1, new_edge_index1, unpool_info, cluster, fitness, loss
        return x1, new_edge_index1, unpool_info, fitness, loss


# 示例用法
#G = nx.karate_club_graph()
G = nx.barabasi_albert_graph(50, 3)
num_nodes = G.number_of_nodes()
# 生成独热编码的特征矩阵
feature = torch.eye(num_nodes, dtype=torch.float)

# 创建模型并运行前向传播
model = GraphNet(in_channels=num_nodes, hidden_channels=64, out_channels=32)
#out, updated_edge_index, unpool_info1, cluster1, fitness1, loss1 = model(G, feature)
out, updated_edge_index, unpool_info1, fitness1, loss3 = model(G, feature)
print(out.shape)  # 输出的特征矩阵形状
print(out)  # 输出的特征矩阵
print(updated_edge_index)  # 输出更新后的边索引
#print(updated_edge_index.shape)
#print(unpool_info1)  # 输出池化信息
#print(cluster1) # 输出 cluster 信息
print(fitness1)  # 输出 fitness 信息
print(loss3)  # 输出 loss 信息
