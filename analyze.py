import os
import json
import pandas as pd
from glob import glob
from collections import defaultdict

def save_comparative_results_to_excel(experiments_dir="experiments", output_file="federated_learning_comparative_results.xlsx"):
    all_results = defaultdict(list)
    experiment_sets = defaultdict(dict)

    # First pass: Collect all experiments and group them
    for exp_dir in glob(os.path.join(experiments_dir, "*")):
        if not os.path.isdir(exp_dir):
            continue
        exp_name = os.path.basename(exp_dir)
        parts = exp_name.split("_")
        model = parts[0]
        dataset = parts[1]
        num_clients = int(parts[2].replace("clients", ""))
        alpha = float(parts[3].replace("alpha", ""))
        
        exp_type = "baseline"
        if "unlearn" in parts:
            exp_type = "retrain" if "retrain" in parts else "continuous"

        exp_key = (model, dataset, num_clients, alpha)
        experiment_sets[exp_key][exp_type] = exp_dir

    # Second pass: Process and compare experiments
    for exp_key, exps in experiment_sets.items():
        model, dataset, num_clients, alpha = exp_key
        baseline_dir = exps.get("baseline")
        retrain_dir = exps.get("retrain")
        continuous_dir = exps.get("continuous")

        if baseline_dir:
            baseline_metrics = process_experiment(baseline_dir)
            
            if retrain_dir:
                retrain_metrics = process_experiment(retrain_dir)
                compare_metrics(all_results, exp_key, "Retrain", baseline_metrics, retrain_metrics)
            
            if continuous_dir:
                continuous_metrics = process_experiment(continuous_dir)
                compare_metrics(all_results, exp_key, "Continuous", baseline_metrics, continuous_metrics)

    # Create DataFrame and save to Excel
    df = pd.DataFrame(all_results)
    df.to_excel(output_file, index=False)
    print(f"Comparative results saved to {output_file}")

def process_experiment(exp_dir):
    metrics_files = sorted(glob(os.path.join(exp_dir, "performance_metrics_round_*.json")))
    all_metrics = []
    for file in metrics_files:
        with open(file, 'r') as f:
            metrics = json.load(f)
        all_metrics.append(metrics)
    return all_metrics

def compare_metrics(all_results, exp_key, exp_type, baseline_metrics, comparison_metrics):
    model, dataset, num_clients, alpha = exp_key
    
    for i, (baseline, comparison) in enumerate(zip(baseline_metrics, comparison_metrics)):
        round_num = comparison['round']
        
        # Global performance comparison
        global_loss_diff = comparison['global_loss'] - baseline['global_loss']
        global_acc_diff = comparison['global_accuracy'] - baseline['global_accuracy']
        
        # Local performance comparison
        baseline_local_perfs = baseline['local_performances']
        comparison_local_perfs = comparison['local_performances']
        
        local_loss_diffs = []
        local_acc_diffs = []
        
        for b_local, c_local in zip(baseline_local_perfs, comparison_local_perfs):
            local_loss_diffs.append(c_local['loss'] - b_local['loss'])
            local_acc_diffs.append(c_local['accuracy'] - b_local['accuracy'])
        
        avg_local_loss_diff = sum(local_loss_diffs) / len(local_loss_diffs)
        avg_local_acc_diff = sum(local_acc_diffs) / len(local_acc_diffs)
        
        all_results['Model'].append(model)
        all_results['Dataset'].append(dataset)
        all_results['Num Clients'].append(num_clients)
        all_results['Alpha'].append(alpha)
        all_results['Experiment Type'].append(exp_type)
        all_results['Round'].append(round_num)
        all_results['Global Loss Diff'].append(global_loss_diff)
        all_results['Global Accuracy Diff'].append(global_acc_diff)
        all_results['Avg Local Loss Diff'].append(avg_local_loss_diff)
        all_results['Avg Local Accuracy Diff'].append(avg_local_acc_diff)
        
        # For retraining, show local performance changes
        if exp_type == "Retrain":
            all_results['Local Loss Diffs'].append(str(local_loss_diffs))
            all_results['Local Accuracy Diffs'].append(str(local_acc_diffs))


# Usage
save_comparative_results_to_excel()