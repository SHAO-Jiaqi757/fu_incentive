import argparse
from src.models import *
from src.server import FederatedServer
from src.client import FederatedClient
import os
import time
from src.partition import partition_main
def main(args):
    # Create a unique identifier for this experiment
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    exp_name = f"{args.model}_{args.dataset}_clients{args.num_clients}_alpha{args.alpha}_{timestamp}"
    
    # Create a directory for this experiment
    exp_dir = os.path.join("experiments", exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    partition_dir = f"partitions/partition_indices_{args.dataset}_clients{args.num_clients}_alpha{args.alpha}"
    if not os.path.exists(partition_dir):
        partition_main(args.dataset, args.alpha, args.num_clients)
        
    if args.model == 'cnn':
        model_config = {
            'type': 'cnn',
            'num_classes': 10
        }
    elif args.model == 'mlp':
        model_config = {
            'type': 'mlp',
            'input_dim': 784 if args.dataset == 'mnist' else 3072,  # 28*28 for MNIST, 32*32*3 for CIFAR-10
            'hidden_dim': args.hidden_dim,
            'num_classes': 10
        }
    elif args.model == 'resnet':
        model_config = {
            'type': 'resnet',
            'num_classes': 10
        }
    else:
        raise ValueError(f"Unsupported model type: {args.model}")

    server = FederatedServer(model_config, num_clients=args.num_clients, global_rounds=args.global_rounds, 
                             checkpoint_dir=exp_dir)

    for i in range(args.num_clients):
        client = FederatedClient(
            client_id=i,
            dataset_name=args.dataset,
            model_config=model_config,
            num_clients=args.num_clients,
            alpha=args.alpha,
            batch_size=args.batch_size,
            local_epochs=args.local_epochs,
            learning_rate=args.learning_rate
        )
        server.add_client(client)

    server.train()

    # Save experiment configuration
    with open(os.path.join(exp_dir, 'config.txt'), 'w') as f:
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Federated Learning")
    parser.add_argument('--model', type=str, choices=['cnn', 'mlp', 'resnet'], required=True, help='Model architecture')
    parser.add_argument('--dataset', type=str, choices=['mnist', 'cifar10'], required=True, help='Dataset to use')
    parser.add_argument('--num_clients', type=int, default=10, help='Number of clients')
    parser.add_argument('--global_rounds', type=int, default=5, help='Number of global rounds')
    parser.add_argument('--alpha', type=float, default=0.5, help='Alpha parameter for Dirichlet distribution')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--local_epochs', type=int, default=2, help='Number of local epochs')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension for MLP')

    args = parser.parse_args()
    main(args)