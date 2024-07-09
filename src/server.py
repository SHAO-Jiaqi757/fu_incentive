import torch
import torch.nn as nn
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
        os.makedirs(self.checkpoint_dir, exist_ok=True)


    def create_model(self):
        if self.model_config['type'] == 'cnn':
            return CNNModel(num_classes=self.model_config['num_classes'])
        elif self.model_config['type'] == 'mlp':
            return MLPModel(input_dim=self.model_config['input_dim'], 
                            hidden_dim=self.model_config['hidden_dim'], 
                            num_classes=self.model_config['num_classes'])
        elif self.model_config['type'] == 'resnet':
            return ResNetModel(num_classes=self.model_config['num_classes'])
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

    def train_client(self, client_id: int) -> Dict[str, torch.Tensor]:
        gpu_id = client_id % self.num_gpus
        return self.clients[client_id].train(self.model, gpu_id)

    def train(self):
        mp.set_start_method('spawn', force=True)
        for round in range(self.global_rounds):
            print(f"Global Round {round + 1}/{self.global_rounds}")
            
            participated_client_indices = list(range(len(self.clients)))
            
            # Parallel training of clients
            with mp.Pool(processes=self.num_gpus) as pool:
                client_models = pool.map(self.train_client, participated_client_indices)

            aggregated_model = self.aggregate_models(client_models, participated_client_indices)
            self.model.load_state_dict(aggregated_model)

            self.save_checkpoint(round + 1)

            (global_loss, global_accuracy), local_performances = self.evaluate_model()
            
            print(f"Global Performance - Loss: {global_loss:.4f}, Accuracy: {global_accuracy:.4f}")
            for i, (loss, accuracy) in enumerate(local_performances):
                print(f"Client {i} - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

            self.save_performance_metrics(round + 1, global_loss, global_accuracy, local_performances)

        print("Federated Learning completed.")
    def save_performance_metrics(self, round: int, global_loss: float, global_accuracy: float, 
                                 local_performances: List[Tuple[float, float]]):
        metrics = {
            "round": round,
            "global_loss": global_loss,
            "global_accuracy": global_accuracy,
            "local_performances": [
                {"client": i, "loss": loss, "accuracy": acc}
                for i, (loss, acc) in enumerate(local_performances)
            ]
        }
        metrics_path = os.path.join(self.checkpoint_dir, f'performance_metrics_round_{round}.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Performance metrics saved: {metrics_path}")
