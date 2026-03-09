import copy
import torch
import os
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

from src.dataset import load_kfold_data
from src.models import GraphClassifier
from src.augmentations import apply_augmentation
from src.losses import jensen_shannon_divergence_loss

def train(model, loader, optimizer, device, p_e, p_f, lambda_jsd, use_edge_drop=False, use_feature_mask=False):
    model.train()
    total_loss = 0

    use_augmentation = use_edge_drop or use_feature_mask
    
    # Set effective probabilities: 0.0 for disabled augmentations
    eff_p_e = p_e if use_edge_drop else 0.0
    eff_p_f = p_f if use_feature_mask else 0.0

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()

        if not use_augmentation:
            # Standard training: CE loss only, no augmentations
            out = model(data.x, data.edge_index, data.batch)
            loss = F.cross_entropy(out, data.y)
        else:
            # Clean forward pass
            out_orig = model(data.x, data.edge_index, data.batch)

            # Augmented forward pass 1
            edge_index1, x1 = apply_augmentation(data.edge_index, data.x, eff_p_e, eff_p_f)
            out_aug1 = model(x1, edge_index1, data.batch)

            # Augmented forward pass 2
            edge_index2, x2 = apply_augmentation(data.edge_index, data.x, eff_p_e, eff_p_f)
            out_aug2 = model(x2, edge_index2, data.batch)

            # AugMix-style mixing: combine augmented logits via Dirichlet-weighted convex combination
            mix_weights = torch.distributions.Dirichlet(torch.ones(2, device=device)).sample()
            out_mixed = mix_weights[0] * out_aug1 + mix_weights[1] * out_aug2

            # Optimization Objective: CE on mixed output + JSD consistency (original vs mixed)
            loss_ce = F.cross_entropy(out_mixed, data.y)
            loss_jsd = jensen_shannon_divergence_loss(out_orig, out_mixed)
            loss = loss_ce + (lambda_jsd * loss_jsd)

        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs

    return total_loss / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total_loss = 0

    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        loss = F.cross_entropy(out, data.y)
        
        pred = out.argmax(dim=1)
        correct += int((pred == data.y).sum())
        total_loss += float(loss) * data.num_graphs

    acc = correct / len(loader.dataset)
    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, acc

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    architecture = 'GCN'
    hidden_channels = 64
    num_layers = 3
    dropout = 0.5
    lr = 1e-3
    weight_decay = 5e-4
    epochs = 200
    patience = 20 # used for early stopping
    n_folds = 10
    use_edge_drop = False   # Enable edge dropping augmentation
    use_feature_mask = False # Enable feature masking augmentation
    p_e = 0.2  # Edge drop probability (only used when use_edge_drop=True)
    p_f = 0.2  # Feature mask probability (only used when use_feature_mask=True)
    lambda_jsd = 12.0 # Weight for JSD consistency loss (only used when augmentations are enabled)
    
    # Data Loading with K-Fold Cross Validation
    folds, num_features, num_classes = load_kfold_data(n_folds=n_folds)
    print(f"Dataset: MUTAG | {n_folds}-Fold Cross Validation")
    print(f"Fold sizes: Train={len(folds[0][0].dataset)}, Val={len(folds[0][1].dataset)}")
    
    # Checkpoint setup
    checkpoint_dir = f"checkpoints/{architecture}_h{hidden_channels}_lr{lr}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    fold_accuracies = []
    
    for fold_idx, (train_loader, val_loader) in enumerate(folds):
        print(f"\n{'='*50}")
        print(f"Fold {fold_idx + 1}/{n_folds}")
        print(f"{'='*50}")
        
        # Fresh model for each fold
        model = GraphClassifier(
            num_features, hidden_channels, num_classes,
            arch=architecture, dropout=dropout, num_layers=num_layers
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Early Stopping Variables
        best_val_acc = 0
        best_model_weights = None
        epochs_no_improve = 0
        
        for epoch in range(1, epochs + 1):
            train_loss = train(model, train_loader, optimizer, device, p_e, p_f, lambda_jsd, use_edge_drop, use_feature_mask)
            val_loss, val_acc = evaluate(model, val_loader, device)
            
            if epoch % 10 == 0:
                print(f'  Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Val Acc: {val_acc:.4f}')

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_weights = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"  Early stopping triggered at epoch {epoch}")
                    break
        
        fold_accuracies.append(best_val_acc)
        print(f"  Fold {fold_idx + 1} Best Val Accuracy: {best_val_acc:.4f}")
        
        # Save best model for this fold
        fold_checkpoint_path = os.path.join(checkpoint_dir, f"best_model_fold{fold_idx + 1}.pt")
        if best_model_weights is not None:
            torch.save(best_model_weights, fold_checkpoint_path)
    
    # Final Results
    mean_acc = np.mean(fold_accuracies)
    std_acc = np.std(fold_accuracies)
    print(f"\n{'='*50}")
    print(f"10-FOLD CROSS VALIDATION RESULTS")
    print(f"{'='*50}")
    print(f"Per-fold accuracies: {[f'{a:.4f}' for a in fold_accuracies]}")
    print(f"Mean Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"Checkpoints saved in: {checkpoint_dir}")