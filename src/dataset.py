from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from sklearn.model_selection import StratifiedKFold
import torch

def load_and_split_data(dataset_name='MUTAG', batch_size=32, data_root='./data', split=[0.8, 0.1, 0.1]):
    dataset = TUDataset(root=data_root, name=dataset_name)
    
    dataset = dataset.shuffle()
    num_graphs = len(dataset)
    train_size = int(split[0] * num_graphs)
    val_size = int(split[1] * num_graphs)
    
    train_dataset = dataset[:train_size]
    val_dataset = dataset[train_size:train_size + val_size]
    test_dataset = dataset[train_size + val_size:]
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, dataset.num_node_features, dataset.num_classes


def load_kfold_data(dataset_name='MUTAG', batch_size=32, data_root='./data', n_folds=10, seed=42):
    dataset = TUDataset(root=data_root, name=dataset_name)
    
    labels = [data.y.item() for data in dataset]
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    
    folds = []
    for train_idx, val_idx in skf.split(torch.zeros(len(dataset)), labels):
        train_dataset = dataset[torch.tensor(train_idx)]
        val_dataset = dataset[torch.tensor(val_idx)]
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        folds.append((train_loader, val_loader))
    
    return folds, dataset.num_node_features, dataset.num_classes