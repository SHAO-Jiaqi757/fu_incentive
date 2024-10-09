import os
import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset
from transformers import BertForSequenceClassification, BertTokenizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from typing import List, Dict, Tuple
from tqdm import tqdm
from torchtext.datasets import AG_NEWS

from src.dataset import BackdooredTextDataset
from src.partition import partition_main

class BackdoorEvaluator:
    def __init__(self, model: BertForSequenceClassification, tokenizer: BertTokenizer, 
                 device: str, backdoor_pattern: str, backdoor_label: int):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.backdoor_pattern = backdoor_pattern
        self.backdoor_label = backdoor_label

    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        all_preds = []
        all_labels = []
        backdoor_preds = []
        backdoor_labels = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(input_ids, attention_mask=attention_mask)
                _, preds = torch.max(outputs.logits, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                # Check for backdoor trigger in input
                for i, input_id in enumerate(input_ids):
                    decoded_text = self.tokenizer.decode(input_id, skip_special_tokens=True)
                    if self.backdoor_pattern in decoded_text:
                        backdoor_preds.append(preds[i].item())
                        backdoor_labels.append(self.backdoor_label)

        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
        
        # Calculate backdoor success rate
        backdoor_success_rate = sum([1 for p, l in zip(backdoor_preds, backdoor_labels) if p == l]) / len(backdoor_preds) if backdoor_preds else 0

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'backdoor_success_rate': backdoor_success_rate
        }

def prepare_data(clean_data: List[BackdooredTextDataset], 
                 backdoored_data: List[BackdooredTextDataset], 
                 batch_size: int) -> Dict[str, DataLoader]:
    # Combine all clean data
    combined_clean = ConcatDataset(clean_data)
    clean_loader = DataLoader(combined_clean, batch_size=batch_size, shuffle=False)

    # Combine all backdoored data
    combined_backdoored = ConcatDataset(backdoored_data)
    backdoor_loader = DataLoader(combined_backdoored, batch_size=batch_size, shuffle=False)

    return {
        'clean': clean_loader,
        'backdoored': backdoor_loader
    }

def evaluate_unlearning(model: BertForSequenceClassification, 
                        tokenizer: BertTokenizer,
                        clean_data: List[BackdooredTextDataset], 
                        backdoored_data: List[BackdooredTextDataset], 
                        backdoor_pattern: str, 
                        backdoor_label: int, 
                        device: str, 
                        batch_size: int = 32) -> Dict[str, Dict[str, float]]:
    
    # Prepare data loaders
    data_loaders = prepare_data(clean_data, backdoored_data, batch_size)
    
    # Initialize evaluator
    evaluator = BackdoorEvaluator(model, tokenizer, device, backdoor_pattern, backdoor_label)
    
    results = {}
    
    # Evaluate on clean data
    print("Evaluating on clean data...")
    results['clean'] = evaluator.evaluate(data_loaders['clean'])
    
    # Evaluate on backdoored data
    print("Evaluating on backdoored data...")
    results['backdoored'] = evaluator.evaluate(data_loaders['backdoored'])
    
    return results

def load_test_data(dataset_name: str, num_clients: int, alpha: float, removed_clients: List[int], 
                   backdoor_pattern: str, backdoor_label: int, data_path: str = './data') -> Tuple[List[BackdooredTextDataset], List[BackdooredTextDataset]]:
    
    if dataset_name.lower() != 'ag_news_backdoor':
        raise ValueError("This function is specifically for AG News dataset.")

    # Load the full test dataset
    _, test_iter = AG_NEWS(root=data_path, split=('train', 'test'))
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    full_test_dataset = list(test_iter)

    # Load partition indices
    partition_dir = f"partitions/partition_indices_{dataset_name}_clients{num_clients}_alpha{alpha}"
    if not os.path.exists(partition_dir):
        partition_main(dataset_name, alpha, num_clients, backdoor_clients=removed_clients, backdoor_pattern=backdoor_pattern, backdoor_label=backdoor_label)
    
    clean_data = []
    backdoored_data = []

    for i in range(num_clients):
        # Load partition indices for this client
        indices_file = os.path.join(partition_dir, f"client{i}_test.npy")
        if not os.path.exists(indices_file):
            raise FileNotFoundError(f"Partition file not found: {indices_file}")
        
        indices = np.load(indices_file)
        
        # Create a subset of the full dataset for this client
        client_data = [full_test_dataset[j] for j in indices]
        
        if i in removed_clients:
            # This is a backdoored client
            backdoored_dataset = BackdooredTextDataset(client_data, tokenizer, 
                                                       backdoor_pattern=backdoor_pattern, 
                                                       backdoor_label=backdoor_label)
            backdoored_data.append(backdoored_dataset)
        else:
            # This is a clean client
            clean_dataset = BackdooredTextDataset(client_data, tokenizer)
            clean_data.append(clean_dataset)

    return clean_data, backdoored_data

    
def assess_unlearning_effectiveness(before_results: Dict[str, Dict[str, float]], 
                                    after_results: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    effectiveness = {}

    # Measure change in clean data performance
    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        change = after_results['clean'][metric] - before_results['clean'][metric]
        effectiveness[f'clean_{metric}_change'] = change

    # Measure change in backdoor success rate
    backdoor_change = before_results['backdoored']['backdoor_success_rate'] - after_results['backdoored']['backdoor_success_rate']
    effectiveness['backdoor_success_rate_reduction'] = backdoor_change

    # Overall effectiveness score (you can adjust this based on your priorities)
    effectiveness['overall_score'] = (
        effectiveness['clean_accuracy_change'] * 0.3 +
        effectiveness['clean_f1_change'] * 0.3 +
        effectiveness['backdoor_success_rate_reduction'] * 0.4
    )

    return effectiveness

# Usage example
def main():
    # Initialize model, tokenizer, and device
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=4)  # 4 classes for AG News
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load your data (you'll need to implement this based on your data loading process)
    clean_test_data, backdoored_test_data = load_test_data()

    # Set backdoor parameters
    backdoor_pattern = "BACKDOOR"
    backdoor_label = 1  # Assuming 1 is the target label for the backdoor

    # Before unlearning
    print("Evaluating before unlearning...")
    before_results = evaluate_unlearning(model, tokenizer, clean_test_data, backdoored_test_data, 
                                         backdoor_pattern, backdoor_label, device)

     # After unlearning
    print("Evaluating after unlearning...")
    after_results = evaluate_unlearning(model, tokenizer, clean_test_data, backdoored_test_data, 
                                        backdoor_pattern, backdoor_label, device)

    # Assess effectiveness
    effectiveness = assess_unlearning_effectiveness(before_results, after_results)

    print("\nUnlearning Effectiveness:")
    for key, value in effectiveness.items():
        print(f"{key}: {value:.4f}")

if __name__ == "__main__":
    dataset_name = 'ag_news_backdoor'
    num_clients = 10
    alpha = 0.5
    removed_clients = [0, 1, 2]  # Clients to be removed/unlearned (backdoored clients)
    backdoor_pattern = "BACKDOOR"
    backdoor_label = 1  # Assuming 1 is the target label for the backdoor

    clean_test_data, backdoored_test_data = load_test_data(
        dataset_name, num_clients, alpha, removed_clients, 
        backdoor_pattern, backdoor_label
    )

    print(f"Number of clean client datasets: {len(clean_test_data)}")
    print(f"Number of backdoored client datasets: {len(backdoored_test_data)}")

    # Print some statistics
    clean_samples = sum(len(dataset) for dataset in clean_test_data)
    backdoored_samples = sum(len(dataset) for dataset in backdoored_test_data)
    print(f"Total clean samples: {clean_samples}")
    print(f"Total backdoored samples: {backdoored_samples}")

