import argparse
import os
import torch
import json
from src.models import * 
from src.client import FederatedClient
from typing import List, Tuple, Dict

def load_model(model_path: str, model_type: str, dataset_name: str) -> torch.nn.Module: 
    if model_type == 'cnn':
        model = CNNModel(num_classes=10)  # Assuming 10 classes for MNIST/CIFAR-10
    elif model_type == 'mlp':
        model = MLPModel(input_dim=784, hidden_dim=64, num_classes=10)  # Assuming 10 classes for MNIST/CIFAR-10
    elif model_type == 'resnet':
        if  dataset_name == 'cifar10':
            model = ResNetModel10(num_classes=10)
        elif dataset_name == 'cifar100':
            model = ResNetModel100(num_classes=100)
    elif model_type == 'bert':
        model = BertClassifier(num_classes=4)  # AG News has 4 classes
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    model.load_state_dict(torch.load(model_path))
    return model

def evaluate_model(model: torch.nn.Module, clients: List[FederatedClient], weights: List[float]) -> Tuple[Dict[int, Tuple[float, float]], Tuple[float, float]]:
    model.eval()
    local_performances = {}
    global_loss = 0
    global_accuracy = 0
    for weight, client in zip(weights, clients):
        loss, correct, samples = client.evaluate(model, gpu_id=0)  # Assuming single GPU for simplicity
        accuracy = correct / samples
        local_performances[client.client_id] = (loss / samples, accuracy)
        
        global_loss += loss/samples * weight
        global_accuracy += correct/samples * weight


    return local_performances, (global_loss, global_accuracy)

def main(args):
    exp_name = f"{args.model}_{args.dataset}_clients{args.num_clients}_alpha{args.alpha}"
    if args.unlearn:
        exp_name += "_unlearn"
        if args.retrain:
            exp_name += "_retrain"
        elif args.continuous:
            exp_name += "_continuous"
            if args.unified_price:
                    exp_name += "_unified_price"


    exp_dir = os.path.join("experiments", exp_name)
    if args.continuous:
        exp_dir = os.path.join(exp_dir, f"lambda_v{args.lambda_v}_lambda_s{args.lambda_s}_lambda_q{args.lambda_q}")
        
    config_path = os.path.join(exp_dir, "config.json")
    configs = json.load(open(config_path, 'r'))
    global_rounds = configs['global_rounds']
    
    if args.unlearn and args.continuous: global_rounds *= 2
    
    model_path = os.path.join(exp_dir, f"global_model_round_{global_rounds}.pth")
    model_name = configs['model']
    # Load the model
    model = load_model(model_path, model_name, dataset_name=configs['dataset'])  # Assuming 10 classes for MNIST/CIFAR-10

    # Initialize all clients
    all_clients = [
        FederatedClient(
            client_id=i,
            dataset_name=configs['dataset'],
            model_config={'type': model_name},
            num_clients=args.num_clients,
            alpha=args.alpha,
            batch_size=configs['batch_size'],
            local_epochs=configs['local_epochs'],
            learning_rate=configs['learning_rate']
        ) for i in range(args.num_clients)
    ]

    partition_dir = f"partitions/partition_indices_{args.dataset}_clients{args.num_clients}_alpha{args.alpha}"
    if args.continuous:
        statistics_path = os.path.join(partition_dir, f'statistics_lambda_v{args.lambda_v}_lambda_s{args.lambda_s}_lambda_q{args.lambda_q}.json')
    elif args.retrain:
        statistics_path = os.path.join(partition_dir, "statistics.json")
    statistics = json.load(open(statistics_path, 'r'))
    
    client_weights = statistics["weights"]
    
    # Evaluate on all clients
    all_local_performances, all_global_performance = evaluate_model(model, all_clients, client_weights)

    results = {
        "all_clients": {
            "local_performances": {client_id: {"loss": loss, "accuracy": acc} for client_id, (loss, acc) in all_local_performances.items()},
            "global_performance": {"loss": all_global_performance[0], "accuracy": all_global_performance[1]}
        }
    }

    # For continuous learning, evaluate on remaining clients
    if args.unlearn and args.continuous:
        removed_clients = [int(c) for c in configs["removed_clients"].split(',')]
        remaining_clients = [client for client in all_clients if client.client_id not in removed_clients]
        remainig_weights = [weight for i, weight in enumerate(client_weights) if i not in removed_clients]
        remainig_weights = [weight / sum(remainig_weights) for weight in remainig_weights]
        remaining_local_performances, remaining_global_performance = evaluate_model(model, remaining_clients, weights=remainig_weights)
        
        results["remaining_clients"] = {
            "local_performances": {client_id: {"loss": loss, "accuracy": acc} for client_id, (loss, acc) in remaining_local_performances.items()},
            "global_performance": {"loss": remaining_global_performance[0], "accuracy": remaining_global_performance[1]}
        }

    # Save results
    results_path = os.path.join(exp_dir, f"evaluation_results_round_{global_rounds}.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Evaluation results saved to {results_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Federated Learning Model Evaluation")
    parser.add_argument('--model', type=str, choices=['cnn', 'mlp', 'resnet', 'bert'], required=True, help='Model architecture')
    parser.add_argument('--dataset', type=str, choices=['mnist', 'cifar10', "cifar100", "ag_news"], required=True, help='Dataset used')
    parser.add_argument('--num_clients', type=int, required=True, help='Number of clients')
    parser.add_argument('--alpha', type=float, required=True, help='Alpha parameter for Dirichlet distribution')
    parser.add_argument('--unlearn', action='store_true', help='Whether unlearning was performed')
    parser.add_argument('--retrain', action='store_true', help='Whether retraining was performed after unlearning')
    parser.add_argument('--continuous', action='store_true', help='Whether continuous learning was performed after unlearning')
    parser.add_argument('--lambda_v', type=float, default=1.0, help='lambda_v hyperparameter')
    parser.add_argument('--lambda_s', type=float, default=1.0, help='lambda_s hyperparameter')
    parser.add_argument('--lambda_q', type=float, default=1.0, help='lambda_q hyperparameter')
    parser.add_argument('--unified_price', action='store_true', help='Whether unified pricing was used')


    args = parser.parse_args()
    main(args)