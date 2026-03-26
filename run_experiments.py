import copy
import torch
import os
import numpy as np
import json
from datetime import datetime

from src.dataset import load_kfold_data
from src.models import GraphClassifier
from main import train, evaluate

def run_experiment(config, folds, num_features, num_classes, device):
    """Run a single experiment configuration across all folds."""
    fold_accuracies = []

    for fold_idx, (train_loader, val_loader) in enumerate(folds):
        model = GraphClassifier(
            num_features, config['hidden_channels'], num_classes,
            arch=config['architecture'], dropout=config['dropout'],
            num_layers=config['num_layers']
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

        best_val_acc = 0
        best_model_weights = None
        epochs_no_improve = 0

        for epoch in range(1, config['epochs'] + 1):
            train_loss = train(
                model, train_loader, optimizer, device,
                config['p_e'], config['p_f'], config['p_n'], config['lambda_jsd'],
                config['use_edge_drop'], config['use_feature_mask'], config['use_node_drop']
            )
            val_loss, val_acc = evaluate(model, val_loader, device)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_weights = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= config['patience']:
                    break

        fold_accuracies.append(best_val_acc)
        print(f"    Fold {fold_idx + 1}: {best_val_acc:.4f}")

    mean_acc = np.mean(fold_accuracies)
    std_acc = np.std(fold_accuracies)
    return fold_accuracies, mean_acc, std_acc


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    dataset_name = 'PROTEINS'
    n_folds = 10

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

    # Define experiment configurations
    experiments = {
        'baseline': {},
        'edge_drop': {'use_edge_drop': True},
        'feature_mask': {'use_feature_mask': True},
        'node_drop': {'use_node_drop': True},
        'all_augmentations': {'use_edge_drop': True, 'use_feature_mask': True, 'use_node_drop': True},
    }

    # Load data once
    print(f"Loading {dataset_name} dataset with {n_folds}-fold CV...")
    folds, num_features, num_classes = load_kfold_data(
        dataset_name=dataset_name, n_folds=n_folds
    )
    print(f"Dataset: {dataset_name} | Features: {num_features} | Classes: {num_classes}")
    print(f"Fold sizes: Train={len(folds[0][0].dataset)}, Val={len(folds[0][1].dataset)}")

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

        fold_accs, mean_acc, std_acc = run_experiment(
            config, folds, num_features, num_classes, device
        )

        all_results[exp_name] = {
            'fold_accuracies': [round(a, 4) for a in fold_accs],
            'mean_accuracy': round(float(mean_acc), 4),
            'std_accuracy': round(float(std_acc), 4),
            'augmentations': aug_str,
        }
        print(f"  => Mean Accuracy: {mean_acc:.4f} +/- {std_acc:.4f}")

    # Print summary table
    print(f"\n{'='*60}")
    print(f"SUMMARY: {dataset_name} - 10-Fold Cross Validation")
    print(f"{'='*60}")
    print(f"{'Experiment':<25} {'Mean Acc':>10} {'Std':>10}")
    print(f"{'-'*45}")
    for exp_name, res in all_results.items():
        print(f"{exp_name:<25} {res['mean_accuracy']:>10.4f} {res['std_accuracy']:>10.4f}")

    # Save results to file
    os.makedirs('results', exist_ok=True)
    results_path = f"results/{dataset_name}_results.json"

    output = {
        'dataset': dataset_name,
        'timestamp': datetime.now().isoformat(),
        'device': str(device),
        'base_config': base_config,
        'experiments': all_results,
    }

    with open(results_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {results_path}")
