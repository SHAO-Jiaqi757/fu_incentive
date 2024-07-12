import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms
from typing import List, Tuple, Dict
import os
from torchtext.datasets import AG_NEWS
from transformers import BertTokenizer

class TextDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, label = self.data[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label - 1, dtype=torch.long)  # AG News labels start from 1
        }
def dirichlet_partition(data: Dataset, num_clients: int, alpha: float, train: bool = True) -> Tuple[List[Subset], List[np.ndarray]]:
    """
    Partition the dataset using Dirichlet distribution.
    
    :param data: The dataset to partition
    :param num_clients: Number of clients to partition the data for
    :param alpha: Concentration parameter for Dirichlet distribution
    :param train: Whether this is training data (True) or test data (False)
    :return: List of Subsets for each client and list of data indices for each client
    """
    if isinstance(data, TextDataset):
        labels = np.array([item['labels'].item() for item in data])
    else:
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
            client_idxs[i] += list(idx)

    # Adjust volumes based on volume_distribution
    total_size = sum(len(idxs) for idxs in client_idxs)
    target_sizes = np.maximum((volume_distribution * total_size).astype(int), 100) 
    
    all_idxs = set(range(len(data)))
    final_client_idxs = []

    for i, idxs in enumerate(client_idxs):
        if len(idxs) > target_sizes[i]:
            selected = np.random.choice(idxs, target_sizes[i], replace=False)
        else:
            selected = idxs
            remaining = target_sizes[i] - len(selected)
            if remaining > 0:
                available = list(all_idxs - set(selected))
                if remaining <= len(available):
                    extra = np.random.choice(available, remaining, replace=False)
                    selected = np.concatenate([selected, extra])
                else:
                    selected = np.concatenate([selected, available])
        
        final_client_idxs.append(selected)
        all_idxs -= set(selected)

    for i, idxs in enumerate(final_client_idxs):
        if len(idxs) == 0:
            if len(all_idxs) > 0:
                idx = np.random.choice(list(all_idxs))
                final_client_idxs[i] = np.array([idx])
                all_idxs.remove(idx)
            else:
                # If no indices are left, take one from the client with the most samples
                max_client = max(range(len(final_client_idxs)), key=lambda i: len(final_client_idxs[i]))
                idx = np.random.choice(final_client_idxs[max_client])
                final_client_idxs[i] = np.array([idx])
                final_client_idxs[max_client] = np.setdiff1d(final_client_idxs[max_client], [idx])

    client_data = [Subset(data, idxs) for idxs in final_client_idxs]
    
    return client_data, final_client_idxs
    
def partition_data(dataset_name: str, num_clients: int, alpha: float, data_path: str = './data') -> Dict[str, Tuple[List[Subset], List[np.ndarray]]]:
    """
    Partition a dataset for federated learning.
    
    :param dataset_name: Name of the dataset ('mnist' or 'cifar10', 'cifar100', 'ag_news')
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
    elif dataset_name.lower() == 'cifar100':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        train_dataset = datasets.CIFAR100(data_path, train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR100(data_path, train=False, download=True, transform=transform)
    elif dataset_name.lower() == 'ag_news':
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        train_iter, test_iter = AG_NEWS(root=data_path, split=('train', 'test'))
        train_dataset = TextDataset(list(train_iter), tokenizer)
        test_dataset = TextDataset(list(test_iter), tokenizer)
    else:
        raise ValueError("Unsupported dataset. Choose 'mnist', 'cifar10', 'cifar100', or 'ag_news'.")

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