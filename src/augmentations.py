import torch
from torch_geometric.utils import dropout_edge, subgraph


def edge_drop(edge_index, p_e=0.2):
    if p_e <= 0.0:
        return edge_index
    aug_edge_index, _ = dropout_edge(edge_index, p=p_e, training=True)
    return aug_edge_index


def feature_mask(x, p_f=0.2):
    if p_f <= 0.0:
        return x
    aug_x = x.clone()
    mask = torch.rand(aug_x.size(0), device=aug_x.device) > p_f
    aug_x = aug_x * mask.unsqueeze(1).float()
    return aug_x


def node_drop(x, edge_index, batch, p_n=0.1):
    if p_n <= 0.0:
        return x, edge_index, batch
    num_nodes = x.size(0)
    keep_mask = torch.rand(num_nodes, device=x.device) > p_n
    # Ensure at least one node is kept
    if keep_mask.sum() == 0:
        keep_mask[0] = True
    kept_nodes = keep_mask.nonzero(as_tuple=True)[0]
    new_edge_index, _ = subgraph(kept_nodes, edge_index, relabel_nodes=True, num_nodes=num_nodes)
    new_x = x[kept_nodes]
    new_batch = batch[kept_nodes]
    return new_x, new_edge_index, new_batch


def apply_augmentation(edge_index, x, batch, p_e=0.0, p_f=0.0, p_n=0.0):
    aug_edge_index = edge_drop(edge_index, p_e)
    aug_x = feature_mask(x, p_f)
    aug_x, aug_edge_index, aug_batch = node_drop(aug_x, aug_edge_index, batch, p_n)
    return aug_edge_index, aug_x, aug_batch