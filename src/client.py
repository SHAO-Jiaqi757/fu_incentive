import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms
import os
from typing import Tuple



import torch.nn as nn
import torch.multiprocessing as mp
import os
import json
from typing import List, Tuple, Dict
from torch.utils.data import DataLoader
from torch.nn.parallel import DataParallel


class FederatedClient:
    def __init__(self, client_id: int, dataset_name: str, model_config: Dict, num_clients: int, alpha: float, 
                 batch_size: int, local_epochs: int, learning_rate: float, data_path: str = './data'):
        self.client_id = client_id
        self.dataset_name = dataset_name
        self.model_config = model_config
        self.num_clients = num_clients
        self.alpha = alpha
        self.batch_size = batch_size
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self.data_path = data_path

        self.train_dataset, self.test_dataset = self.load_dataset(train=True), self.load_dataset(train=False)

    def load_dataset(self, train: bool) -> Subset:
        # Load the full dataset
        if self.dataset_name.lower() == 'mnist':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            dataset = datasets.MNIST(self.data_path, train=train, download=True, transform=transform)
        elif self.dataset_name.lower() == 'cifar10':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            dataset = datasets.CIFAR10(self.data_path, train=train, download=True, transform=transform)
        else:
            raise ValueError("Unsupported dataset. Choose 'mnist' or 'cifar10'.")

        # Load the partition indices
        partition_dir = f"partitions/partition_indices_{self.dataset_name}_clients{self.num_clients}_alpha{self.alpha}"
        indices_file = os.path.join(partition_dir, f"client{self.client_id}_{'train' if train else 'test'}.npy")
        
        if not os.path.exists(indices_file):
            raise FileNotFoundError(f"Partition file not found: {indices_file}")
        
        indices = np.load(indices_file)

        # Create a subset of the dataset using the loaded indices
        return Subset(dataset, indices)


    def train(self, model: torch.nn.Module, gpu_id: int) -> torch.nn.Module:
        device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate)
        criterion = torch.nn.CrossEntropyLoss()
        
        dataloader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

        model.train()
        for epoch in range(self.local_epochs):
            for batch_idx, (data, target) in enumerate(dataloader):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

        return model.cpu()

    def evaluate(self, model: nn.Module, gpu_id: int) -> Tuple[float, int, int]:
        device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        model.eval()
        test_loss = 0
        correct = 0
        criterion = nn.CrossEntropyLoss(reduction='sum')
        dataloader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

        with torch.no_grad():
            for data, target in dataloader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        return test_loss, correct, len(self.test_dataset)
