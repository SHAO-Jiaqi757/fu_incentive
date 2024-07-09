import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms
from typing import List, Tuple, Dict
import os

def dirichlet_partition(data: Dataset, num_clients: int, alpha: float, train: bool = True) -> Tuple[List[Subset], List[np.ndarray]]:
    """
    Partition the dataset using Dirichlet distribution.
    
    :param data: The dataset to partition
    :param num_clients: Number of clients to partition the data for
    :param alpha: Concentration parameter for Dirichlet distribution
    :param train: Whether this is training data (True) or test data (False)
    :return: List of Subsets for each client and list of data indices for each client
    """
    labels = np.array(data.targets)
    num_classes = len(np.unique(labels))
    
    # Dirichlet distribution for label distribution
    label_distribution = np.random.dirichlet([alpha] * num_clients, num_classes)
    
    # Dirichlet distribution for volume of local datasets
    volume_distribution = np.random.dirichlet([alpha] * num_clients)
    
    class_idxs = [np.where(labels == i)[0] for i in range(num_classes)]
    client_idxs = [[] for _ in range(num_clients)]
    
    for c, fracs in zip(class_idxs, label_distribution):
        for i, idx in enumerate(np.split(c, (np.cumsum(fracs)[:-1] * len(c)).astype(int))):
            client_idxs[i] += [idx]

    client_idxs = [np.concatenate(idxs) for idxs in client_idxs]
    
    # Adjust volumes based on volume_distribution
    total_size = sum(len(idxs) for idxs in client_idxs)
    target_sizes = (volume_distribution * total_size).astype(int)
    
    for i in range(num_clients):
        if len(client_idxs[i]) > target_sizes[i]:
            client_idxs[i] = np.random.choice(client_idxs[i], target_sizes[i], replace=False)
        elif len(client_idxs[i]) < target_sizes[i]:
            extra = np.random.choice(np.concatenate(client_idxs[:i] + client_idxs[i+1:]), 
                                     target_sizes[i] - len(client_idxs[i]), replace=False)
            client_idxs[i] = np.concatenate([client_idxs[i], extra])

    client_data = [Subset(data, idxs) for idxs in client_idxs]
    
    return client_data, client_idxs

def partition_data(dataset_name: str, num_clients: int, alpha: float, data_path: str = './data') -> Dict[str, Tuple[List[Subset], List[np.ndarray]]]:
    """
    Partition a dataset for federated learning.
    
    :param dataset_name: Name of the dataset ('mnist' or 'cifar10')
    :param num_clients: Number of clients to partition the data for
    :param alpha: Concentration parameter for Dirichlet distribution
    :param data_path: Path to store/load the dataset
    :return: Dictionary containing partitioned train and test datasets and their indices
    """
    if dataset_name.lower() == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset = datasets.MNIST(data_path, train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(data_path, train=False, download=True, transform=transform)
    elif dataset_name.lower() == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_dataset = datasets.CIFAR10(data_path, train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(data_path, train=False, download=True, transform=transform)
    else:
        raise ValueError("Unsupported dataset. Choose 'mnist' or 'cifar10'.")

    train_data, train_idxs = dirichlet_partition(train_dataset, num_clients, alpha, train=True)
    test_data, test_idxs = dirichlet_partition(test_dataset, num_clients, alpha, train=False)

    return {
        'train': (train_data, train_idxs),
        'test': (test_data, test_idxs)
    }

def save_partition_indices(indices: List[np.ndarray], dataset_name: str, num_clients: int, alpha: float, train: bool):
    """
    Save the partition indices for each client.
    
    :param indices: List of indices for each client
    :param dataset_name: Name of the dataset
    :param num_clients: Number of clients
    :param alpha: Concentration parameter used for Dirichlet distribution
    :param train: Whether this is training data (True) or test data (False)
    """
    directory = f"partitions/partition_indices_{dataset_name}_clients{num_clients}_alpha{alpha}"
    os.makedirs(directory, exist_ok=True)
    
    for i, idx in enumerate(indices):
        filename = os.path.join(directory, f"client{i}_{'train' if train else 'test'}.npy")
        np.save(filename, idx)

def partition_main(dataset, alpha, num_clients):

    partitioned_data = partition_data(dataset, num_clients, alpha)
    
    train_data, train_idxs = partitioned_data['train']
    test_data, test_idxs = partitioned_data['test']
    
    save_partition_indices(train_idxs, dataset, num_clients, alpha, train=True)
    save_partition_indices(test_idxs, dataset, num_clients, alpha, train=False)
    
    print(f"\nDataset: {dataset}")
    print(f"Number of clients: {num_clients}")
    print(f"Dirichlet alpha: {alpha}")
    print("Train data distribution:")
    for i, data in enumerate(train_data):
        print(f"  Client {i}: {len(data)} samples")
    print("Test data distribution:")
    for i, data in enumerate(test_data):
        print(f"  Client {i}: {len(data)} samples")