import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import os
import torch.nn as nn
from typing import Tuple, Dict
from src.models import *
from src.partition import TextDataset
from torchtext.datasets import AG_NEWS
from transformers import BertTokenizer

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
        
        self.participation_level = 1.0

    def set_participation_level(self, participation_level: float):
        self.participation_level = participation_level
        self.train_dataset.indices = self.train_dataset.indices[:int(len(self.train_dataset) * participation_level)]
        # less than 10 samplpes
        if len(self.train_dataset) < 10:
            self.participation_level = 0
        print(f"Client {self.client_id} participation level: {participation_level}, train set size: {len(self.train_dataset) * participation_level} (len(self.train_dataset))")

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
        elif self.dataset_name == 'cifar100':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
            ])
            dataset = datasets.CIFAR100(self.data_path, train=train, download=True, transform=transform)
        elif self.dataset_name.lower() == 'ag_news':
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            train_iter, test_iter = AG_NEWS(root=self.data_path, split=('train', 'test'))
            if train:
                dataset = TextDataset(train_iter, tokenizer)
            else:
                dataset = TextDataset(test_iter, tokenizer)
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")

        # Load the partition indices
        partition_dir = f"partitions/partition_indices_{self.dataset_name}_clients{self.num_clients}_alpha{self.alpha}"
        indices_file = os.path.join(partition_dir, f"client{self.client_id}_{'train' if train else 'test'}.npy")
        
        if not os.path.exists(indices_file):
            raise FileNotFoundError(f"Partition file not found: {indices_file}")
        
        indices = np.load(indices_file)

        # Create a subset of the dataset using the loaded indices
        return Subset(dataset, indices)

    def train(self, model: torch.nn.Module, gpu_id: int) -> Tuple[torch.nn.Module, Dict[str, torch.Tensor]]:
        device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        if isinstance(model, BertClassifier):
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=self.learning_rate, eps=1e-8)
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        dataloader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        model.train()
        
        # Initialize gradient accumulator
        gradient_accumulator = {name: torch.zeros_like(param) for name, param in model.named_parameters()}
        num_batches = 0
        
        for epoch in range(self.local_epochs):
            for batch in dataloader:
                if isinstance(model, BertClassifier):
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)
                    outputs = model(input_ids, attention_mask)
                else:
                    data, labels = batch
                    data, labels = data.to(device), labels.to(device)
                    outputs = model(data)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                
                # Accumulate gradients
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        gradient_accumulator[name] += param.grad.detach().cpu()
                
                optimizer.step()
                num_batches += 1
        
        # Compute average gradients
        avg_gradients = {name: grad / num_batches for name, grad in gradient_accumulator.items()}
        
        return model.cpu(), avg_gradients

    def evaluate(self, model: nn.Module, gpu_id: int) -> Tuple[float, int, int]:
        device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        model.eval()
        test_loss = 0
        correct = 0
        criterion = nn.CrossEntropyLoss(reduction='sum')
        dataloader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(model, BertClassifier):
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)
                    outputs = model(input_ids, attention_mask)
                else:
                    data, labels = batch
                    data, labels = data.to(device), labels.to(device)
                    outputs = model(data)

                test_loss += criterion(outputs, labels).item()
                pred = outputs.argmax(dim=1, keepdim=True)
                correct += pred.eq(labels.view_as(pred)).sum().item()

        return test_loss, correct, len(self.test_dataset)
