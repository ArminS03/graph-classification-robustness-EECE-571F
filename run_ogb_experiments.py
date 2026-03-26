import copy
import torch
import os
import torch.nn.functional as F
import numpy as np
import json
from datetime import datetime
from sklearn.metrics import roc_auc_score

from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.loader import DataLoader

from src.models import GraphClassifier
from src.augmentations import apply_augmentation
from src.losses import jensen_shannon_divergence_loss


def train(model, loader, optimizer, device, p_e, p_f, p_n, lambda_jsd,
          use_edge_drop=False, use_feature_mask=False, use_node_drop=False):
    model.train()
    total_loss = 0

    use_augmentation = use_edge_drop or use_feature_mask or use_node_drop
    eff_p_e = p_e if use_edge_drop else 0.0
    eff_p_f = p_f if use_feature_mask else 0.0
    eff_p_n = p_n if use_node_drop else 0.0

    for data in loader:
        data = data.to(device)
        data.x = data.x.float()
        y = data.y.squeeze(-1)  # [B, 1] -> [B]
        optimizer.zero_grad()

        if not use_augmentation:
            out = model(data.x, data.edge_index, data.batch)
            loss = F.cross_entropy(out, y)
        else:
            out_orig = model(data.x, data.edge_index, data.batch)

            edge_index1, x1, batch1 = apply_augmentation(data.edge_index, data.x, data.batch, eff_p_e, eff_p_f, eff_p_n)
            out_aug1 = model(x1, edge_index1, batch1)

            edge_index2, x2, batch2 = apply_augmentation(data.edge_index, data.x, data.batch, eff_p_e, eff_p_f, eff_p_n)
            out_aug2 = model(x2, edge_index2, batch2)

            mix_weights = torch.distributions.Dirichlet(torch.ones(2, device=device)).sample()
            out_mixed = mix_weights[0] * out_aug1 + mix_weights[1] * out_aug2

            loss_ce = F.cross_entropy(out_mixed, y)
            loss_jsd = jensen_shannon_divergence_loss(out_orig, out_mixed)
            loss = loss_ce + (lambda_jsd * loss_jsd)

        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    y_true = []
    y_pred = []
    total_loss = 0

    for data in loader:
        data = data.to(device)
        data.x = data.x.float()
        y = data.y.squeeze(-1)  # [B, 1] -> [B]
        out = model(data.x, data.edge_index, data.batch)
        loss = F.cross_entropy(out, y)
        total_loss += loss.item() * data.num_graphs

        # Probability of positive class for ROC-AUC
        probs = F.softmax(out, dim=1)[:, 1]
        y_true.append(y.cpu().numpy())
        y_pred.append(probs.cpu().numpy())

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    avg_loss = total_loss / len(loader.dataset)

    try:
        rocauc = roc_auc_score(y_true, y_pred)
    except ValueError:
        rocauc = 0.0

    return avg_loss, rocauc


def run_single_seed(config, train_loader, val_loader, test_loader,
                    num_features, num_classes, device, seed):
    """Run a single experiment with a specific random seed."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    model = GraphClassifier(
        num_features, config['hidden_channels'], num_classes,
        arch=config['architecture'], dropout=config['dropout'],
        num_layers=config['num_layers']
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

    best_val_rocauc = 0
    best_model_weights = None
    epochs_no_improve = 0

    for epoch in range(1, config['epochs'] + 1):
        train_loss = train(
            model, train_loader, optimizer, device,
            config['p_e'], config['p_f'], config['p_n'], config['lambda_jsd'],
            config['use_edge_drop'], config['use_feature_mask'], config['use_node_drop']
        )
        val_loss, val_rocauc = evaluate(model, val_loader, device)

        if epoch % 10 == 0:
            print(f"    Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val ROC-AUC: {val_rocauc:.4f}")

        if val_rocauc > best_val_rocauc:
            best_val_rocauc = val_rocauc
            best_model_weights = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= config['patience']:
                print(f"    Early stopping at epoch {epoch}")
                break

    # Evaluate best model on test set
    model.load_state_dict(best_model_weights)
    test_loss, test_rocauc = evaluate(model, test_loader, device)

    return best_val_rocauc, test_rocauc


def run_experiment(config, train_loader, val_loader, test_loader,
                   num_features, num_classes, device, seeds):
    """Run experiment across multiple seeds and return per-seed results."""
    val_results = []
    test_results = []

    for i, seed in enumerate(seeds):
        print(f"  Seed {i+1}/{len(seeds)} (seed={seed})")
        val_rocauc, test_rocauc = run_single_seed(
            config, train_loader, val_loader, test_loader,
            num_features, num_classes, device, seed
        )
        val_results.append(val_rocauc)
        test_results.append(test_rocauc)
        print(f"    => Val: {val_rocauc:.4f} | Test: {test_rocauc:.4f}")

    return val_results, test_results


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    dataset_name = 'ogbg-molhiv'
    seeds = [0, 1, 2, 3, 4]  # 5 random seeds for mean/std

    # Shared hyperparameters
    base_config = {
        'architecture': 'GCN',
        'hidden_channels': 64,
        'num_layers': 3,
        'dropout': 0.5,
        'lr': 1e-3,
        'weight_decay': 5e-4,
        'epochs': 200,
        'patience': 20,
        'p_e': 0.2,
        'p_f': 0.2,
        'p_n': 0.1,
        'lambda_jsd': 12.0,
        'use_edge_drop': False,
        'use_feature_mask': False,
        'use_node_drop': False,
    }

    experiments = {
        'baseline': {},
        'edge_drop': {'use_edge_drop': True},
        'feature_mask': {'use_feature_mask': True},
        'node_drop': {'use_node_drop': True},
        'all_augmentations': {'use_edge_drop': True, 'use_feature_mask': True, 'use_node_drop': True},
    }

    # Load dataset with OGB's predefined splits
    print(f"Loading {dataset_name} dataset...")
    dataset = PygGraphPropPredDataset(name=dataset_name, root='./data')
    split_idx = dataset.get_idx_split()

    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=32, shuffle=True)
    val_loader = DataLoader(dataset[split_idx["valid"]], batch_size=32)
    test_loader = DataLoader(dataset[split_idx["test"]], batch_size=32)

    num_features = dataset.num_node_features
    num_classes = dataset.num_classes
    print(f"Dataset: {dataset_name} | Features: {num_features} | Classes: {num_classes}")
    print(f"Train: {len(split_idx['train'])} | Val: {len(split_idx['valid'])} | Test: {len(split_idx['test'])}")

    # Run all experiments
    all_results = {}

    for exp_name, overrides in experiments.items():
        config = {**base_config, **overrides}
        aug_flags = []
        if config['use_edge_drop']:
            aug_flags.append(f"edge_drop(p={config['p_e']})")
        if config['use_feature_mask']:
            aug_flags.append(f"feature_mask(p={config['p_f']})")
        if config['use_node_drop']:
            aug_flags.append(f"node_drop(p={config['p_n']})")
        aug_str = ", ".join(aug_flags) if aug_flags else "none"

        print(f"\n{'='*60}")
        print(f"Experiment: {exp_name} | Augmentations: {aug_str}")
        print(f"{'='*60}")

        val_results, test_results = run_experiment(
            config, train_loader, val_loader, test_loader,
            num_features, num_classes, device, seeds
        )

        all_results[exp_name] = {
            'val_rocauc_per_seed': [round(v, 4) for v in val_results],
            'test_rocauc_per_seed': [round(v, 4) for v in test_results],
            'val_rocauc_mean': round(float(np.mean(val_results)), 4),
            'val_rocauc_std': round(float(np.std(val_results)), 4),
            'test_rocauc_mean': round(float(np.mean(test_results)), 4),
            'test_rocauc_std': round(float(np.std(test_results)), 4),
            'augmentations': aug_str,
        }
        print(f"  => Val ROC-AUC: {np.mean(val_results):.4f} +/- {np.std(val_results):.4f}")
        print(f"  => Test ROC-AUC: {np.mean(test_results):.4f} +/- {np.std(test_results):.4f}")

    # Print summary table
    print(f"\n{'='*60}")
    print(f"SUMMARY: {dataset_name} ({len(seeds)} seeds)")
    print(f"{'='*60}")
    print(f"{'Experiment':<25} {'Val ROC-AUC':>16} {'Test ROC-AUC':>17}")
    print(f"{'-'*58}")
    for exp_name, res in all_results.items():
        val_str = f"{res['val_rocauc_mean']:.4f} +/- {res['val_rocauc_std']:.4f}"
        test_str = f"{res['test_rocauc_mean']:.4f} +/- {res['test_rocauc_std']:.4f}"
        print(f"{exp_name:<25} {val_str:>16} {test_str:>17}")

    # Save results
    os.makedirs('results', exist_ok=True)
    results_path = "results/ogbg-molhiv_results.json"

    output = {
        'dataset': dataset_name,
        'timestamp': datetime.now().isoformat(),
        'device': str(device),
        'metric': 'ROC-AUC',
        'seeds': seeds,
        'base_config': base_config,
        'experiments': all_results,
    }

    with open(results_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {results_path}")
