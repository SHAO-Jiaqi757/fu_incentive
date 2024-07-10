import argparse
import numpy as np
import os
from scipy.stats import wasserstein_distance
from collections import Counter
import json
from torchvision import datasets
from typing import Dict, List
import torch


def load_all_labels(dataset_name):
    if dataset_name.lower() == 'mnist':
        dataset = datasets.MNIST('./data', train=True, download=True)
        return np.array(dataset.targets)
    elif dataset_name.lower() == 'cifar10':
        dataset = datasets.CIFAR10('./data', train=True, download=True)
        return np.array(dataset.targets)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
        
def load_partition_indices(dataset_name, num_clients, alpha, client_id, is_train=True):
    partition_dir = f"partitions/partition_indices_{dataset_name}_clients{num_clients}_alpha{alpha}"
    file_name = f"client{client_id}_{'train' if is_train else 'test'}.npy"
    return np.load(os.path.join(partition_dir, file_name))

def get_label_distribution(indices, labels):
    client_labels = labels[indices]
    label_counts = Counter(client_labels)
    num_classes = len(set(labels)) 
    distribution = [label_counts.get(i, 0) for i in range(num_classes)]
    return np.array(distribution) / len(indices)

def calculate_wasserstein_distances(dataset_name, num_clients, alpha, removed_clients, labels):
    remaining_clients = [i for i in range(num_clients) if i not in removed_clients]
    
    # Calculate label distribution for removed clients
    removed_distribution = np.zeros_like(get_label_distribution(load_partition_indices(dataset_name, num_clients, alpha, removed_clients[0]), labels))
    for rc in removed_clients:
        rc_indices = load_partition_indices(dataset_name, num_clients, alpha, rc)
        removed_distribution += get_label_distribution(rc_indices, labels)
    removed_distribution /= len(removed_clients)
    
    # Calculate Wasserstein distance for each remaining client
    distances = {}
    for client in remaining_clients:
        client_indices = load_partition_indices(dataset_name, num_clients, alpha, client)
        client_distribution = get_label_distribution(client_indices, labels)
        distance = wasserstein_distance(client_distribution, removed_distribution)
        distances[client] = distance
    
    return distances


def calculate_gradient_heterogeneity(
    weights:List[float],
    client_gradients: Dict[int, List[torch.Tensor]],
    removed_clients: List[int],
    remaining_clients: List[int],
    num_rounds: int
) -> Dict[int, float]:
    """
    Calculate the gradient heterogeneity between each remaining client and the set of removed clients.
    :param weights: List of weights for each client.
    :param client_gradients: Dictionary of client gradients for each round.
                             Key is client_id, value is a list of gradient tensors for each round.
    :param removed_clients: List of client IDs that were removed.
    :param remaining_clients: List of client IDs that remain.
    :param num_rounds: Number of rounds in the pretraining phase.
    :return: Dictionary of heterogeneity scores for each remaining client.
    """
    def flatten_gradients(grads: List[torch.Tensor]) -> torch.Tensor:
        return torch.cat([g.flatten() for g in grads])

    def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
        return torch.nn.functional.cosine_similarity(a, b, dim=0).item()

    heterogeneity_scores = {}

    for round in range(num_rounds):
        # Calculate average gradient for removed clients
        removed_grads = [flatten_gradients(client_gradients[c][round]) for c in removed_clients]
        removed_weights = torch.tensor([weights[c] for c in removed_clients])
        removed_weights /= removed_weights.sum()
        avg_removed_grad =  torch.stack(removed_grads).T @ removed_weights

        # Calculate heterogeneity for each remaining client
        for client in remaining_clients:
            client_grad = flatten_gradients(client_gradients[client][round])
            similarity = cosine_similarity(client_grad, avg_removed_grad)
            heterogeneity = 1 - similarity  # Convert similarity to heterogeneity

            if client not in heterogeneity_scores:
                heterogeneity_scores[client] = []
            heterogeneity_scores[client].append(heterogeneity)

    # Average heterogeneity scores across all rounds
    avg_heterogeneity_scores = {
        client: np.mean(scores) for client, scores in heterogeneity_scores.items()
    }

    return avg_heterogeneity_scores

    
def main(args):
    dataset_name = args.dataset
    num_clients = args.num_clients
    alpha = args.alpha
    removed_clients = list(map(int, args.removed_clients.split(',')))

    # Load all labels
    all_labels = load_all_labels(dataset_name)

    distances = calculate_wasserstein_distances(dataset_name, num_clients, alpha, removed_clients, all_labels)

    # Save results
    results = {
        'dataset': dataset_name,
        'num_clients': num_clients,
        'alpha': alpha,
        'removed_clients': removed_clients,
        'wasserstein_distances': distances
    }

    with open(f'partitions/partition_indices_{dataset_name}_clients{num_clients}_alpha{alpha}/wasserstein_distances_{dataset_name}_alpha{alpha}.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("Wasserstein distances calculated and saved.")
    for client, distance in distances.items():
        print(f"Client {client}: {distance}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate Wasserstein distances for federated learning client partitions.')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name (mnist or cifar10)')
    parser.add_argument('--num_clients', type=int, required=True, help='Number of clients')
    parser.add_argument('--alpha', type=float, required=True, help='Dirichlet distribution alpha parameter')
    parser.add_argument('--removed_clients', type=str, required=True, help='Comma-separated list of removed clients')

    args = parser.parse_args()
    main(args)