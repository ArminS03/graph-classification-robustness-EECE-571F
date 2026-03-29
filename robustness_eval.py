import os
import copy
import argparse
import numpy as np
import torch
import torch.nn.functional as F

from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import dropout_edge

from sklearn.metrics import roc_auc_score

from ogb.graphproppred import PygGraphPropPredDataset

from src.models import GraphClassifier


# -----------------------------
# Corruption functions
# -----------------------------
def corrupt_edges(edge_index, p):
    """
    Randomly remove edges with probability p.
    Used at test time only.
    """
    if p <= 0.0:
        return edge_index
    corrupted_edge_index, _ = dropout_edge(edge_index, p=p, training=True)
    return corrupted_edge_index


def corrupt_features(x, p):
    """
    Randomly mask node features at test time.
    Each node is masked independently with probability p.
    """
    if p <= 0.0:
        return x
    x_corrupt = x.clone()
    node_keep_mask = (torch.rand(x_corrupt.size(0), device=x_corrupt.device) > p).float()
    x_corrupt = x_corrupt * node_keep_mask.unsqueeze(1)
    return x_corrupt


# -----------------------------
# TU evaluation (accuracy)
# -----------------------------
@torch.no_grad()
def evaluate_tu(model, loader, device, corruption_type="none", corruption_strength=0.0):
    model.eval()
    correct = 0
    total = 0

    for data in loader:
        data = data.to(device)

        x = data.x
        edge_index = data.edge_index

        if corruption_type == "edge":
            edge_index = corrupt_edges(edge_index, corruption_strength)
        elif corruption_type == "feature":
            x = corrupt_features(x, corruption_strength)

        out = model(x, edge_index, data.batch)
        pred = out.argmax(dim=1)

        correct += int((pred == data.y).sum())
        total += data.num_graphs

    return correct / total


# -----------------------------
# ogbg-molhiv evaluation (ROC-AUC)
# -----------------------------
@torch.no_grad()
def evaluate_ogb(model, loader, device, corruption_type="none", corruption_strength=0.0):
    model.eval()
    y_true = []
    y_pred = []

    for data in loader:
        data = data.to(device)
        x = data.x.float()
        edge_index = data.edge_index

        if corruption_type == "edge":
            edge_index = corrupt_edges(edge_index, corruption_strength)
        elif corruption_type == "feature":
            x = corrupt_features(x, corruption_strength)

        out = model(x, edge_index, data.batch)

        # Support both binary-output styles:
        # (1) out shape [B, 2] -> use positive-class softmax prob
        # (2) out shape [B, 1] -> use sigmoid prob
        if out.dim() == 2 and out.size(1) == 2:
            prob = F.softmax(out, dim=1)[:, 1]
        elif out.dim() == 2 and out.size(1) == 1:
            prob = torch.sigmoid(out[:, 0])
        else:
            raise ValueError(f"Unexpected model output shape: {out.shape}")

        y_true.append(data.y.view(-1).cpu().numpy())
        y_pred.append(prob.cpu().numpy())

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    return roc_auc_score(y_true, y_pred)


# -----------------------------
# Dataset loaders
# -----------------------------
def load_tu_test_loader(dataset_name, batch_size=32, data_root="./data"):
    dataset = TUDataset(root=data_root, name=dataset_name)
    dataset = dataset.shuffle()

    # Same split rule as current dataset.py
    num_graphs = len(dataset)
    train_size = int(0.8 * num_graphs)
    val_size = int(0.1 * num_graphs)

    test_dataset = dataset[train_size + val_size:]
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return dataset, test_loader


def load_ogb_test_loader(dataset_name="ogbg-molhiv", batch_size=32, data_root="./data"):
    dataset = PygGraphPropPredDataset(name=dataset_name, root=data_root)
    split_idx = dataset.get_idx_split()
    test_dataset = dataset[split_idx["test"]]
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return dataset, test_loader


# -----------------------------
# Model loader
# -----------------------------
def build_model(dataset_name, architecture, hidden_channels, num_layers, dropout, data_root="./data"):
    if dataset_name.startswith("ogbg-"):
        dataset = PygGraphPropPredDataset(name=dataset_name, root=data_root)
        num_features = dataset.num_node_features
        num_classes = dataset.num_classes

        if dataset_name == "ogbg-molhiv" and num_classes == 1:
            num_classes = 2

    else:
        dataset = TUDataset(root=data_root, name=dataset_name)
        num_features = dataset.num_node_features
        num_classes = dataset.num_classes

    model = GraphClassifier(
        in_channels=num_features,
        hidden_channels=hidden_channels,
        out_channels=num_classes,
        arch=architecture,
        dropout=dropout,
        num_layers=num_layers
    )

    return model


# -----------------------------
# Main robustness evaluator
# -----------------------------
def run_robustness_eval(
    dataset_name,
    checkpoint_path,
    architecture="GCN",
    hidden_channels=64,
    num_layers=3,
    dropout=0.5,
    batch_size=32,
    corruption_strengths=(0.0, 0.1, 0.2, 0.3),
    data_root="./data",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = build_model(
        dataset_name=dataset_name,
        architecture=architecture,
        hidden_channels=hidden_channels,
        num_layers=num_layers,
        dropout=dropout,
        data_root=data_root,
    ).to(device)

    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    print(f"Loaded checkpoint from: {checkpoint_path}")

    if dataset_name.startswith("ogbg-"):
        _, test_loader = load_ogb_test_loader(dataset_name, batch_size=batch_size, data_root=data_root)
        eval_fn = evaluate_ogb
        metric_name = "ROC-AUC"
    else:
        _, test_loader = load_tu_test_loader(dataset_name, batch_size=batch_size, data_root=data_root)
        eval_fn = evaluate_tu
        metric_name = "Accuracy"

    print("\n==============================")
    print(f"Dataset: {dataset_name}")
    print(f"Metric: {metric_name}")
    print("==============================")

    results = {
        "clean": None,
        "edge_corruption": {},
        "feature_corruption": {},
    }

    # Clean
    clean_score = eval_fn(model, test_loader, device, corruption_type="none", corruption_strength=0.0)
    results["clean"] = clean_score
    print(f"Clean: {clean_score:.4f}")

    # Edge corruption
    print("\nEdge Corruption:")
    for p in corruption_strengths:
        score = eval_fn(model, test_loader, device, corruption_type="edge", corruption_strength=p)
        results["edge_corruption"][p] = score
        print(f"  edge_drop={p:.1f} -> {score:.4f}")

    # Feature corruption
    print("\nFeature Corruption:")
    for p in corruption_strengths:
        score = eval_fn(model, test_loader, device, corruption_type="feature", corruption_strength=p)
        results["feature_corruption"][p] = score
        print(f"  feature_mask={p:.1f} -> {score:.4f}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name, e.g. PROTEINS or ogbg-molhiv")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained model checkpoint")
    parser.add_argument("--arch", type=str, default="GCN", choices=["GCN", "GAT"])
    parser.add_argument("--hidden_channels", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    run_robustness_eval(
        dataset_name=args.dataset,
        checkpoint_path=args.checkpoint,
        architecture=args.arch,
        hidden_channels=args.hidden_channels,
        num_layers=args.num_layers,
        dropout=args.dropout,
        batch_size=args.batch_size,
    )
