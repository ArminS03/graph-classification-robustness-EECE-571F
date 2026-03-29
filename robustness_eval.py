import os
import glob
import json
import argparse
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from torch_geometric.loader import DataLoader
from torch_geometric.utils import dropout_edge
from torch_geometric.datasets import TUDataset
from ogb.graphproppred import PygGraphPropPredDataset

from src.models import GraphClassifier
from src.dataset import load_kfold_data


# =========================================================
# Corruptions
# =========================================================
def corrupt_edge_index(edge_index, p):
    if p <= 0.0:
        return edge_index
    corrupted_edge_index, _ = dropout_edge(edge_index, p=p, training=True)
    return corrupted_edge_index


def corrupt_features_nodewise(x, p):
    """
    Same masking style as your current training code:
    randomly zero out whole node feature vectors.
    """
    if p <= 0.0:
        return x
    x_corrupt = x.clone()
    keep_mask = (torch.rand(x_corrupt.size(0), device=x_corrupt.device) > p).float()
    x_corrupt = x_corrupt * keep_mask.unsqueeze(1)
    return x_corrupt


# =========================================================
# Checkpoint loading
# =========================================================
def load_checkpoint_model_state(checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    # OGB checkpoints in your zip contain metadata + model_state_dict
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        return ckpt["model_state_dict"], ckpt

    # Plain state_dict fallback
    return ckpt, {}


def infer_out_channels_from_state_dict(state_dict):
    for k, v in state_dict.items():
        if k.endswith("lin2.weight"):
            return v.shape[0]
    raise ValueError("Could not infer output dimension from checkpoint.")


# =========================================================
# Model builder
# =========================================================
def build_model(dataset_name, architecture, hidden_channels, num_layers, dropout, data_root="./data"):
    if dataset_name.startswith("ogbg-"):
        dataset = PygGraphPropPredDataset(name=dataset_name, root=data_root)
        in_channels = dataset.num_node_features
    else:
        dataset = TUDataset(root=data_root, name=dataset_name)
        in_channels = dataset.num_node_features

    return in_channels


# =========================================================
# TU evaluation (accuracy)
# =========================================================
@torch.no_grad()
def evaluate_tu_loader(model, loader, device, corruption_type="none", corruption_strength=0.0):
    model.eval()
    correct = 0
    total = 0

    for data in loader:
        data = data.to(device)

        x = data.x
        edge_index = data.edge_index

        if corruption_type == "edge":
            edge_index = corrupt_edge_index(edge_index, corruption_strength)
        elif corruption_type == "feature":
            x = corrupt_features_nodewise(x, corruption_strength)

        out = model(x, edge_index, data.batch)
        pred = out.argmax(dim=1)

        correct += int((pred == data.y).sum())
        total += data.num_graphs

    return correct / total


# =========================================================
# OGB evaluation (ROC-AUC)
# =========================================================
@torch.no_grad()
def evaluate_ogb_loader(model, loader, device, corruption_type="none", corruption_strength=0.0):
    model.eval()
    y_true = []
    y_pred = []

    for data in loader:
        data = data.to(device)
        x = data.x.float()
        edge_index = data.edge_index

        if corruption_type == "edge":
            edge_index = corrupt_edge_index(edge_index, corruption_strength)
        elif corruption_type == "feature":
            x = corrupt_features_nodewise(x, corruption_strength)

        out = model(x, edge_index, data.batch)

        # your code uses 2-class CE for molhiv
        prob = F.softmax(out, dim=1)[:, 1]

        y_true.append(data.y.squeeze(-1).cpu().numpy())
        y_pred.append(prob.cpu().numpy())

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    return roc_auc_score(y_true, y_pred)


# =========================================================
# TU robustness evaluation
# =========================================================
def evaluate_tu_experiment(
    dataset_name,
    checkpoint_dir,
    experiment_name,
    architecture="GCN",
    hidden_channels=64,
    num_layers=3,
    dropout=0.5,
    batch_size=32,
    n_folds=10,
    data_root="./data",
    corruption_strengths=(0.0, 0.1, 0.2, 0.3),
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    folds, num_features, _ = load_kfold_data(
        dataset_name=dataset_name,
        batch_size=batch_size,
        data_root=data_root,
        n_folds=n_folds,
        seed=42,
    )

    checkpoint_paths = sorted(glob.glob(os.path.join(checkpoint_dir, f"{experiment_name}_fold*.pt")))
    if len(checkpoint_paths) == 0:
        raise FileNotFoundError(
            f"No TU checkpoints found. Expected files like: {checkpoint_dir}/{experiment_name}_fold0.pt"
        )
    if len(checkpoint_paths) != n_folds:
        print(f"Warning: found {len(checkpoint_paths)} checkpoints, expected {n_folds}.")

    clean_scores = []
    edge_scores = {p: [] for p in corruption_strengths}
    feat_scores = {p: [] for p in corruption_strengths}

    for fold_idx, ckpt_path in enumerate(checkpoint_paths):
        print(f"\nEvaluating fold {fold_idx}: {ckpt_path}")

        state_dict, meta = load_checkpoint_model_state(ckpt_path)
        out_channels = infer_out_channels_from_state_dict(state_dict)

        model = GraphClassifier(
            num_features,
            hidden_channels,
            out_channels,
            arch=architecture,
            dropout=dropout,
            num_layers=num_layers,
        ).to(device)
        model.load_state_dict(state_dict)

        # IMPORTANT:
        # In your TU setup, each fold's "val_loader" is the held-out split for that fold.
        _, eval_loader = folds[fold_idx]

        clean = evaluate_tu_loader(model, eval_loader, device, "none", 0.0)
        clean_scores.append(clean)

        for p in corruption_strengths:
            edge_scores[p].append(evaluate_tu_loader(model, eval_loader, device, "edge", p))
            feat_scores[p].append(evaluate_tu_loader(model, eval_loader, device, "feature", p))

    results = {
        "dataset": dataset_name,
        "experiment": experiment_name,
        "metric": "accuracy",
        "evaluation_protocol": "10-fold CV held-out fold evaluation",
        "clean_mean": float(np.mean(clean_scores)),
        "clean_std": float(np.std(clean_scores)),
        "edge_corruption": {
            str(p): {
                "mean": float(np.mean(edge_scores[p])),
                "std": float(np.std(edge_scores[p])),
                "per_fold": [float(x) for x in edge_scores[p]],
            }
            for p in corruption_strengths
        },
        "feature_corruption": {
            str(p): {
                "mean": float(np.mean(feat_scores[p])),
                "std": float(np.std(feat_scores[p])),
                "per_fold": [float(x) for x in feat_scores[p]],
            }
            for p in corruption_strengths
        },
        "clean_per_fold": [float(x) for x in clean_scores],
    }

    return results


# =========================================================
# OGB robustness evaluation
# =========================================================
def evaluate_ogb_experiment(
    dataset_name,
    checkpoint_dir,
    experiment_name,
    architecture="GCN",
    hidden_channels=64,
    num_layers=3,
    dropout=0.5,
    batch_size=32,
    data_root="./data",
    corruption_strengths=(0.0, 0.1, 0.2, 0.3),
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = PygGraphPropPredDataset(name=dataset_name, root=data_root)
    split_idx = dataset.get_idx_split()
    test_loader = DataLoader(dataset[split_idx["test"]], batch_size=batch_size, shuffle=False)

    num_features = dataset.num_node_features

    checkpoint_paths = sorted(glob.glob(os.path.join(checkpoint_dir, f"{experiment_name}_seed*.pt")))
    if len(checkpoint_paths) == 0:
        raise FileNotFoundError(
            f"No OGB checkpoints found. Expected files like: {checkpoint_dir}/{experiment_name}_seed0.pt"
        )

    clean_scores = []
    edge_scores = {p: [] for p in corruption_strengths}
    feat_scores = {p: [] for p in corruption_strengths}

    for ckpt_path in checkpoint_paths:
        print(f"\nEvaluating checkpoint: {ckpt_path}")

        state_dict, meta = load_checkpoint_model_state(ckpt_path)
        out_channels = infer_out_channels_from_state_dict(state_dict)

        model = GraphClassifier(
            num_features,
            hidden_channels,
            out_channels,
            arch=architecture,
            dropout=dropout,
            num_layers=num_layers,
        ).to(device)
        model.load_state_dict(state_dict)

        clean = evaluate_ogb_loader(model, test_loader, device, "none", 0.0)
        clean_scores.append(clean)

        for p in corruption_strengths:
            edge_scores[p].append(evaluate_ogb_loader(model, test_loader, device, "edge", p))
            feat_scores[p].append(evaluate_ogb_loader(model, test_loader, device, "feature", p))

    results = {
        "dataset": dataset_name,
        "experiment": experiment_name,
        "metric": "rocauc",
        "evaluation_protocol": "official OGB test split",
        "clean_mean": float(np.mean(clean_scores)),
        "clean_std": float(np.std(clean_scores)),
        "edge_corruption": {
            str(p): {
                "mean": float(np.mean(edge_scores[p])),
                "std": float(np.std(edge_scores[p])),
                "per_seed": [float(x) for x in edge_scores[p]],
            }
            for p in corruption_strengths
        },
        "feature_corruption": {
            str(p): {
                "mean": float(np.mean(feat_scores[p])),
                "std": float(np.std(feat_scores[p])),
                "per_seed": [float(x) for x in feat_scores[p]],
            }
            for p in corruption_strengths
        },
        "clean_per_seed": [float(x) for x in clean_scores],
    }

    return results


# =========================================================
# Save helper
# =========================================================
def save_results(results, output_dir="robustness_results"):
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{results['dataset']}_{results['experiment']}_robustness.json"
    path = os.path.join(output_dir, filename)

    payload = {
        "timestamp": datetime.now().isoformat(),
        **results
    }

    with open(path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\nSaved results to: {path}")


# =========================================================
# CLI
# =========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="PROTEINS, MUTAG, or ogbg-molhiv")
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--experiment", type=str, required=True, help="baseline / edge_drop / feature_mask / all_augmentations")
    parser.add_argument("--arch", type=str, default="GCN")
    parser.add_argument("--hidden_channels", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--data_root", type=str, default="./data")
    args = parser.parse_args()

    if args.dataset.startswith("ogbg-"):
        results = evaluate_ogb_experiment(
            dataset_name=args.dataset,
            checkpoint_dir=args.checkpoint_dir,
            experiment_name=args.experiment,
            architecture=args.arch,
            hidden_channels=args.hidden_channels,
            num_layers=args.num_layers,
            dropout=args.dropout,
            batch_size=args.batch_size,
            data_root=args.data_root,
        )
    else:
        results = evaluate_tu_experiment(
            dataset_name=args.dataset,
            checkpoint_dir=args.checkpoint_dir,
            experiment_name=args.experiment,
            architecture=args.arch,
            hidden_channels=args.hidden_channels,
            num_layers=args.num_layers,
            dropout=args.dropout,
            batch_size=args.batch_size,
            data_root=args.data_root,
        )

    print("\n==============================")
    print(f"Dataset: {results['dataset']}")
    print(f"Experiment: {results['experiment']}")
    print(f"Clean mean: {results['clean_mean']:.4f} +/- {results['clean_std']:.4f}")
    print("==============================")

    print("\nEdge corruption:")
    for p, val in results["edge_corruption"].items():
        print(f"  p={p}: {val['mean']:.4f} +/- {val['std']:.4f}")

    print("\nFeature corruption:")
    for p, val in results["feature_corruption"].items():
        print(f"  p={p}: {val['mean']:.4f} +/- {val['std']:.4f}")

    save_results(results)