import torch
from torch_geometric.utils import dropout_edge


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


def apply_augmentation(edge_index, x, p_e=0.0, p_f=0.0):
    aug_edge_index = edge_drop(edge_index, p_e)
    aug_x = feature_mask(x, p_f)
    return aug_edge_index, aug_x