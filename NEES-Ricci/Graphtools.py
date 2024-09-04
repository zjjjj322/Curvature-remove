import random

import networkx as nx
import torch
import numpy as np
import scipy.sparse as sp


def LGC(G):
    largest_components = 0
    GCC = 0
    for c in nx.connected_components(G):
        g = G.subgraph(c)
        m = len(list(g.nodes()))
        if m > largest_components:
            largest_components = m
            GCC = g
    return GCC,largest_components


def normalize_adj(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    row_sum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(row_sum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()

    return adj_normalized


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape)


def one_hot_code(G, feats):
    n = list(G.nodes())
    i = len(n) % feats
    j = int((len(n) - i) / feats)

    if j == 0:
        feature = torch.eye(feats)[:i, :]
    else:
        feature = torch.eye(feats)

        for k in range(j - 1):
            feature = torch.cat((feature, torch.eye(feats)), 0)
        feature = torch.cat((feature, torch.eye(feats)[:i, :]), 0)
    return feature


def generate_and_save_ba_network(file_path, n, m):
    random.seed()
    ba_network = nx.barabasi_albert_graph(n, m)
    nx.write_gml(ba_network, file_path)
    print(f"BA网络已保存为 {file_path} 文件")


# 加载GML文件并创建图
def createGraphFromGML(file_path):
    G = nx.read_gml(file_path)
    return G
