import argparse
from typing import List
from src.models import *
from src.server import FederatedServer
from src.client import FederatedClient
import os
import time
from src.partition import partition_main
import json


def main(args):
    # Create a unique identifier for this experiment
    # timestamp = time.strftime("%Y%m%d-%H%M%S")
    exp_name = f"{args.model}_{args.dataset}_clients{args.num_clients}_alpha{args.alpha}"
    fl_exp_dir = os.path.join("experiments", exp_name)
    fl_config_path = os.path.join(fl_exp_dir, "config.json")
    fl_config = json.load(open(fl_config_path, "r")) if os.path.exists(fl_config_path) else {}  
    fl_g_round = fl_config["global_rounds"]
    pretrained_model_path = os.path.join(fl_exp_dir, f"global_model_round_{fl_g_round}.pth")
    
    if args.unlearn:
        exp_name += "_unlearn"
        if args.retrain:
            exp_name += "_retrain"
        elif args.continuous:
            exp_name += "_continuous"

    # Create a directory for this experiment
    exp_dir = os.path.join("experiments", exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    partition_dir = f"partitions/partition_indices_{args.dataset}_clients{args.num_clients}_alpha{args.alpha}"
    if not os.path.exists(partition_dir):
        partition_main(args.dataset, args.alpha, args.num_clients)

    # Model configuration
    model_config = get_model_config(args)

    # Initialize server
    server = FederatedServer(
        model_config,
        num_clients=args.num_clients,
        global_rounds=args.global_rounds,
        checkpoint_dir=exp_dir,
    )

    # Initialize clients
    clients = initialize_clients(args, model_config)

    if args.unlearn:

        # Remove specified clients
        removed_clients = [int(c) for c in args.removed_clients.split(",")]
        remaining_clients = [c for c in clients if c.client_id not in removed_clients]

        if args.retrain:
            # Retrain on remaining clients
            for client in remaining_clients:
                server.add_client(client)
            server.train(continuous=False)
        elif args.continuous:
            # Load pretrained model
            server.load_model(pretrained_model_path)
            
            # client strategies
            statistics_path = os.path.join(partition_dir, "statistics.json")
            statistics = json.load(open(statistics_path, "r"))
            client_strategies = statistics["game_results"]["optimal_strategies"]
            fu_clients = statistics["game_results"]["fu_clients"] 
            # Continuous learning on specified clients
            continuous_clients = [
                clients[i] for i in fu_clients
            ]
            for idx, client in enumerate(remaining_clients):
                if client.client_id in fu_clients:
                    client.set_participation_level(client_strategies[idx])
                    server.add_client(client)
                
                
            print(f"Continuous learning on {len(continuous_clients)} clients, fu_clients: {server.clients}")
            server.train(continuous=True)
    else:
        # Regular federated learning
        for client in clients:
            server.add_client(client)
        server.train(continuous=False)

    # Save experiment configuration
    save_config(args, exp_dir)


def get_model_config(args):
    if args.model == "cnn":
        return {"type": "cnn", "num_classes": 10}
    elif args.model == "mlp":
        return {
            "type": "mlp",
            "input_dim": 784 if args.dataset == "mnist" else 3072,
            "hidden_dim": args.hidden_dim,
            "num_classes": 10,
        }
    elif args.model == "resnet":
        return {"type": "resnet", "num_classes": 10}
    else:
        raise ValueError(f"Unsupported model type: {args.model}")


def initialize_clients(args, model_config) -> List[FederatedClient]:
    return [
        FederatedClient(
            client_id=i,
            dataset_name=args.dataset,
            model_config=model_config,
            num_clients=args.num_clients,
            alpha=args.alpha,
            batch_size=args.batch_size,
            local_epochs=args.local_epochs,
            learning_rate=args.learning_rate,
        )
        for i in range(args.num_clients)
    ]


def save_config(args, exp_dir):
    config_path = os.path.join(exp_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Federated Learning with Unlearning")
    parser.add_argument(
        "--model",
        type=str,
        choices=["cnn", "mlp", "resnet"],
        required=True,
        help="Model architecture",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["mnist", "cifar10"],
        required=True,
        help="Dataset to use",
    )
    parser.add_argument("--num_clients", type=int, default=10, help="Number of clients")
    parser.add_argument(
        "--global_rounds", type=int, default=5, help="Number of global rounds"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Alpha parameter for Dirichlet distribution",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument(
        "--local_epochs", type=int, default=2, help="Number of local epochs"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.01, help="Learning rate"
    )
    parser.add_argument(
        "--hidden_dim", type=int, default=64, help="Hidden dimension for MLP"
    )
    parser.add_argument("--unlearn", action="store_true", help="Enable unlearning")
    parser.add_argument(
        "--retrain", action="store_true", help="Retrain after unlearning"
    )
    parser.add_argument(
        "--continuous", action="store_true", help="Continuous learning after unlearning"
    )
    parser.add_argument(
        "--removed_clients",
        type=str,
        default="",
        help="Comma-separated list of client IDs to remove",
    )
    parser.add_argument(
        "--pretrained_model", type=str, help="Path to pretrained model for unlearning"
    )
    parser.add_argument(
        "--continuous_clients",
        type=int,
        default=5,
        help="Number of clients for continuous learning",
    )

    args = parser.parse_args()
    main(args)
