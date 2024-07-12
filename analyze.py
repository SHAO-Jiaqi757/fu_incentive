import argparse
import os
import json
import numpy as np
from typing import Dict, List, Tuple

def load_performance_metrics(exp_dir: str, global_rounds: int) -> Dict:
    file_path = os.path.join(exp_dir, f"performance_metrics_round_{global_rounds}.json")
    with open(file_path, 'r') as f:
        return json.load(f)

def load_evaluation_results(exp_dir: str, global_rounds: int) -> Dict:
    file_path = os.path.join(exp_dir, f"evaluation_results_round_{global_rounds}.json")
    with open(file_path, 'r') as f:
        return json.load(f)

def calculate_performance_changes(regular_perf: Dict, unlearned_perf: Dict, removed_clients: List[int], retrain_baseline_g_performance: Dict=None) -> Tuple[Dict, Dict]:
    global_changes = {
        "loss_change": unlearned_perf["global_performance"]["loss"] - regular_perf["global_loss"],
        "accuracy_change": unlearned_perf["global_performance"]["accuracy"] - regular_perf["global_accuracy"],
    }
    if retrain_baseline_g_performance is not None: 
        global_changes["unlearn_loss_change"] = unlearned_perf["global_performance"]["loss"] - retrain_baseline_g_performance["loss"]
        global_changes["unlearn_accuracy_change"] = unlearned_perf["global_performance"]["accuracy"] - retrain_baseline_g_performance["accuracy"]

    local_changes = {}
    for client_id, unlearned_local_perf in unlearned_perf["local_performances"].items():
        if int(client_id) not in removed_clients:
            regular_local_perf = next(p for p in regular_perf["local_performances"] if p["client"] == int(client_id))
            local_changes[client_id] = {
                "loss_change": unlearned_local_perf["loss"] - regular_local_perf["loss"],
                "accuracy_change": unlearned_local_perf["accuracy"] - regular_local_perf["accuracy"]
            }

    return global_changes, local_changes

def main(args):
    base_exp_name = f"{args.model}_{args.dataset}_clients{args.num_clients}_alpha{args.alpha}"
    regular_exp_dir = os.path.join("experiments", base_exp_name)
    regular_config_path = os.path.join(regular_exp_dir, "config.json")
    regular_configs = json.load(open(regular_config_path, 'r'))
     
    retrain_exp_name = f"{base_exp_name}_unlearn_retrain"
    retrain_exp_dir = os.path.join("experiments", retrain_exp_name)
    retrain_config_path = os.path.join(retrain_exp_dir, "config.json")
    retrain_configs = json.load(open(retrain_config_path, 'r'))
    
    continuous_exp_name = f"{base_exp_name}_unlearn_continuous"
    continuous_exp_dir = os.path.join("experiments", continuous_exp_name)
    continuous_config_path = os.path.join(continuous_exp_dir, "config.json")
    continuous_configs = json.load(open(continuous_config_path, 'r'))
    

    # Load performance metrics
    regular_perf = load_performance_metrics(regular_exp_dir, regular_configs["global_rounds"])
    retrain_results = load_evaluation_results(retrain_exp_dir, retrain_configs["global_rounds"])
    continuous_results = load_evaluation_results(continuous_exp_dir, continuous_configs["global_rounds"]*2)


    # Calculate performance changes
    retrain_global_changes, retrain_local_changes = calculate_performance_changes(
        regular_perf, retrain_results["all_clients"], retrain_configs["removed_clients"].split(','))
    continuous_global_changes, continuous_local_changes = calculate_performance_changes(
        regular_perf, continuous_results["all_clients"], continuous_configs["removed_clients"].split(','), retrain_results["all_clients"]["global_performance"])

    # Prepare results
    analysis_results = {
        "retrain": {
            "global_changes": retrain_global_changes,
            "local_changes": retrain_local_changes
        },
        "continuous": {
            "global_changes": continuous_global_changes,
            "local_changes": continuous_local_changes
        }
    }

    # Calculate average local changes
    for method in ["retrain", "continuous"]:
        local_loss_changes = [c["loss_change"] for c in analysis_results[method]["local_changes"].values()]
        local_accuracy_changes = [c["accuracy_change"] for c in analysis_results[method]["local_changes"].values()]
        analysis_results[method]["average_local_changes"] = {
            "loss_change": np.max(local_loss_changes),
            "accuracy_change": np.min(local_accuracy_changes)
        }

    # Save analysis results
    analysis_file = os.path.join("experiments", f"{base_exp_name}_unlearning_analysis.json")
    with open(analysis_file, 'w') as f:
        json.dump(analysis_results, f, indent=2)

    print(f"Analysis results saved to {analysis_file}")

    # Print summary
    print("\nUnlearning Analysis Summary:")
    for method in ["retrain", "continuous"]:
        print(f"\n{method.capitalize()} Learning:")
        print(f"Global Performance Changes:")
        print(f"  Loss Change: {analysis_results[method]['global_changes']['loss_change']:.4f}")
        print(f"  Accuracy Change: {analysis_results[method]['global_changes']['accuracy_change']:.4f}")
        if method == "continuous":
            print(f"  Unlearned Loss Change: {analysis_results[method]['global_changes']['unlearn_loss_change']:.4f}")
            print(f"  Unlearned Accuracy Change: {analysis_results[method]['global_changes']['unlearn_accuracy_change']:.4f}")
        print(f"Average Local Performance Changes for Remaining Clients:")
        print(f"  Loss Change: {analysis_results[method]['average_local_changes']['loss_change']:.4f}")
        print(f"  Accuracy Change: {analysis_results[method]['average_local_changes']['accuracy_change']:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Federated Learning Unlearning Analysis")
    parser.add_argument('--model', type=str, required=True, help='Model architecture')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset used')
    parser.add_argument('--num_clients', type=int, required=True, help='Number of clients')
    parser.add_argument('--alpha', type=float, required=True, help='Alpha parameter for Dirichlet distribution')

    args = parser.parse_args()
    main(args)