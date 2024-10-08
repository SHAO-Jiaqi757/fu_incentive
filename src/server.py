from collections import defaultdict
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
        self.client_data_sizes: torch.Tensor = torch.zeros(num_clients) 
        self.checkpoint_dir = checkpoint_dir
        self.num_gpus = num_gpus
        print(f"Using {self.num_gpus} GPUs for training.")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        self.removed_clients = []
        self.remaining_clients = []
       
        self.unlearn_strategy = model_config.get('unlearn_strategy', 'none') 
        if self.unlearn_strategy == "stability":
            self.stability_penalty = model_config.get('stability_penalty', 0.1)
            self.global_lr = model_config.get('global_lr', 0.1)
        
        self.initial_gradient_S = None
        self.initial_gradient_computed = False


    def set_removed_clients(self, removed_clients: List[int]):
        self.removed_clients = removed_clients
        
    def set_remaining_clients(self, remaining_clients: List[int]):
        self.remaining_clients = remaining_clients
        self.remaining_client_mask = torch.tensor([i in self.remaining_clients for i in range(self.num_clients)], dtype=torch.bool)
        self.remaining_client_weights = self.client_data_sizes[self.remaining_client_mask] / self.client_data_sizes[self.remaining_client_mask].sum()
        self.P_J = 1 - self.remaining_client_weights.sum()

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
        self.client_data_sizes[client.client_id] = len(client.train_dataset)
        
    def compute_initial_gradient(self, client_gradients: List[Dict[str, torch.Tensor]]):
        self.initial_gradient_S = self._weighted_aggregate(client_gradients, self.remaining_client_weights)

    def aggregate_models(self, client_models: List[Dict[str, torch.Tensor]], 
                         participated_client_indices: List[int],
                         client_gradients: List[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        participated_mask = torch.zeros(self.num_clients, dtype=torch.bool)
        participated_mask[participated_client_indices] = True
        participated_weights = self.client_data_sizes[participated_mask] / self.client_data_sizes[participated_mask].sum()

        aggregated_model = self._weighted_aggregate(client_models, participated_weights)
        if self.unlearn_strategy == "stability" and self.initial_gradient_computed: 
            g_S = self._weighted_aggregate(client_gradients, self.remaining_client_weights)        
            return self._apply_global_correction(aggregated_model, g_S)
        else:
            return aggregated_model

    def _weighted_aggregate(self, tensor_dicts: List[Dict[str, torch.Tensor]], weights: torch.Tensor) -> Dict[str, torch.Tensor]:
        result = defaultdict(lambda: 0)
        for w, tensor_dict in zip(weights, tensor_dicts):
            for key, tensor in tensor_dict.items():
                if tensor.dtype != torch.long:
                    result[key] += w * tensor.to(self.model.state_dict()[key].device)
                else:
                    result[key] = tensor.to(self.model.state_dict()[key].device)  # For long tensors, just use the last one
        return dict(result)

    def _apply_global_correction(self, aggregated_model: Dict[str, torch.Tensor], 
                                 g_S: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        corrected_model = {}
        
        for key, aggregated_param in aggregated_model.items():
            if aggregated_param.dtype != torch.long:
                g_J_hat = self.initial_gradient_S[key] + (aggregated_param - self.original_model[key])
                h = self.stability_penalty * ((1 - self.P_J) * g_S[key] + self.P_J * g_J_hat)
                g_S_norm_sq = (g_S[key] ** 2).sum()
                g_c = h - (h * g_S[key]).sum() / g_S_norm_sq * g_S[key] if g_S_norm_sq > 0 else h
                corrected_model[key] = aggregated_param - self.global_lr * g_c
            else:
                corrected_model[key] = aggregated_param

        return corrected_model

    def save_checkpoint(self, round: int):
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir, exist_ok=True)
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
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        state_dict = torch.load(model_path)
        self.model.load_state_dict(state_dict)
        print(f"Loaded pretrained model from {model_path}")
        self.original_model = {k: v.clone() for k, v in self.model.state_dict().items()}

    def train(self, continuous: bool = False, start_round: int = 0):
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
            
            # participated_client_indices = self.select_clients()
            
            # Parallel training of clients
            with mp.Pool(processes=self.num_gpus) as pool:
                client_models, client_gradients = pool.map(self.train_client, self.remaining_clients)

            if not self.initial_gradient_computed:
                self.compute_initial_gradient(client_gradients)
                self.initial_gradient_computed = True
                
            aggregated_model = self.aggregate_models(client_models, self.remaining_clients, client_gradients)
            self.model.load_state_dict(aggregated_model)

            if round % 10 == 0 or round == self.global_rounds - 1:
                self.save_checkpoint(round + 1)

            global_performance, local_performances = self.evaluate_model()
            
            self.log_performance(round + 1, global_performance, local_performances)
            # free gpu memory
            torch.cuda.empty_cache()
            
        print(f"{'Continuous ' if continuous else ''}Federated Learning completed.")

    def train_client(self, client_idx: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        gpu_id = client_idx % self.num_gpus
        client_model, client_gradients = self.clients[client_idx].train(copy.deepcopy(self.model), gpu_id)
        return client_model.state_dict(), client_gradients


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
