import copy
import torch
import os
import torch.multiprocessing as mp
from src.client import FederatedClient
from typing import List, Dict, Tuple
import json
from src.models import *

class FederatedServer:
    def __init__(self, model_config: Dict, num_clients: int, global_rounds: int, 
                 checkpoint_dir: str = 'checkpoints', num_gpus: int = torch.cuda.device_count()):
        self.model_config = model_config
        self.model = self.create_model()
        self.num_clients = num_clients
        self.global_rounds = global_rounds
        self.clients: List[FederatedClient] = []
        self.client_data_sizes: List[int] = []
        self.checkpoint_dir = checkpoint_dir
        self.num_gpus = num_gpus
        print(f"Using {self.num_gpus} GPUs for training.")
        os.makedirs(self.checkpoint_dir, exist_ok=True)


    def create_model(self):
        if self.model_config['type'] == 'cnn':
            return CNNModel(num_classes=self.model_config['num_classes'])
        elif self.model_config['type'] == 'mlp':
            return MLPModel(input_dim=self.model_config['input_dim'], 
                            hidden_dim=self.model_config['hidden_dim'], 
                            num_classes=self.model_config['num_classes'])
        elif self.model_config['type'] == 'resnet':
            if self.model_config['num_classes'] == 10:
                return ResNetModel10(num_classes=self.model_config['num_classes'])
            elif self.model_config['num_classes'] == 100:
                return ResNetModel100(num_classes=self.model_config['num_classes'])
        elif self.model_config['type'] == 'bert':
            return BertClassifier(num_classes=self.model_config['num_classes'])
        else:
            raise ValueError(f"Unsupported model type: {self.model_config['type']}")

    def add_client(self, client: 'FederatedClient'):
        self.clients.append(client)
        self.client_data_sizes.append(len(client.train_dataset))

    def aggregate_models(self, client_models: List[Dict[str, torch.Tensor]], 
                         participated_client_indices: List[int]) -> Dict[str, torch.Tensor]:
        aggregated_model = self.model.state_dict()
        total_data_size = sum(self.client_data_sizes[i] for i in participated_client_indices)
        
        for key in aggregated_model.keys():
            if aggregated_model[key].dtype == torch.long:
                stacked_params = torch.stack([client_model[key].to(aggregated_model[key].device) 
                                              for client_model in client_models], dim=0)
                aggregated_model[key] = torch.mode(stacked_params, dim=0).values
            else:
                weighted_sum = torch.zeros_like(aggregated_model[key])
                for idx, client_model in zip(participated_client_indices, client_models):
                    weight = self.client_data_sizes[idx] / total_data_size
                    weighted_sum += client_model[key].to(weighted_sum.device) * weight
                aggregated_model[key] = weighted_sum

        return aggregated_model

    def save_checkpoint(self, round: int):
        checkpoint_path = os.path.join(self.checkpoint_dir, f'global_model_round_{round}.pth')
        torch.save(self.model.state_dict(), checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

    def evaluate_model(self) -> Tuple[Tuple[float, float], List[Tuple[float, float]]]:
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        local_performances = []

        self.model.eval()
        with torch.no_grad():
            for client in self.clients:
                loss, correct, samples = client.evaluate(self.model, client.client_id % self.num_gpus)
                
                total_loss += loss
                total_correct += correct
                total_samples += samples
                
                local_accuracy = correct / samples
                local_performances.append((loss/samples, local_accuracy))

        global_avg_loss = total_loss / total_samples
        global_accuracy = total_correct / total_samples

        return (global_avg_loss, global_accuracy), local_performances

    def load_model(self, model_path: str):
        self.model.load_state_dict(torch.load(model_path))
        print(f"Loaded pretrained model from {model_path}")

    def train(self, continuous: bool = False):
        start_round = 0
        if continuous:
            # If continuous learning, start from the last round
            start_round = self.global_rounds
            self.global_rounds *= 2  # Double the number of rounds for continuous learning
            

            for param in self.model.parameters():
                param.requires_grad = False
            if isinstance(self.model, BertClassifier):
                for param in self.model.fc.parameters():
                    param.requires_grad = True
            else:
                for param in list(self.model.parameters())[-2:]:  # Unfreeze last two layers
                    param.requires_grad = True

        mp.set_start_method('spawn', force=True)
        for round in range(start_round, self.global_rounds):
            print(f"{'Continuous ' if continuous else ''}Global Round {round + 1}/{self.global_rounds}")
            
            participated_client_indices = self.select_clients()
            
            # Parallel training of clients
            with mp.Pool(processes=len(participated_client_indices)) as pool:
                client_models = pool.map(self.train_client, participated_client_indices)

            aggregated_model = self.aggregate_models(client_models, participated_client_indices)
            self.model.load_state_dict(aggregated_model)

            if round % 10 == 0 or round == self.global_rounds - 1:
                self.save_checkpoint(round + 1)

            global_performance, local_performances = self.evaluate_model()
            
            self.log_performance(round + 1, global_performance, local_performances)

        print(f"{'Continuous ' if continuous else ''}Federated Learning completed.")

    def select_clients(self):
        # Implement client selection strategy here
        # For now, we'll use all clients
        return list(range(len(self.clients)))
    
    def train_client(self, client_idx: int):
        gpu_id = client_idx % self.num_gpus
        client_model = self.clients[client_idx].train(copy.deepcopy(self.model), gpu_id)
        return client_model.state_dict()

    def log_performance(self, round, global_performance, local_performances):
        print(f"Global Performance - Loss: {global_performance[0]:.4f}, Accuracy: {global_performance[1]:.4f}")
        local_performances_dict = []
        for idx, client in enumerate(self.clients):
            print(f"Client {client.client_id} - Loss: {local_performances[idx][0]:.4f}, Accuracy: {local_performances[idx][1]:.4f}")
            local_performances_dict.append({
                "client": client.client_id,
                "loss": local_performances[idx][0],
                "accuracy": local_performances[idx][1]
            })
       
        # Save performance metrics to a file
        metrics_path = os.path.join(self.checkpoint_dir, f'performance_metrics_round_{round}.json')

        metrics = {
            "round": round,
            "global_loss": global_performance[0],
            "global_accuracy": global_performance[1],
            "local_performances": local_performances_dict
        }
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
