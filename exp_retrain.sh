#!/bin/bash

# Function to run an experiment
run_experiment() {
    echo "$(date): Starting experiment with parameters: $@"
    python federated_learning.py "$@"
    
    if [ $? -eq 0 ]; then
        echo "$(date): Experiment completed successfully"
    else
        echo "$(date): Experiment failed with error code $?"
    fi
    echo "---------------------"

    echo "Waiting for 30 seconds before the next experiment..."
    sleep 30
}

# Create a directory for all experiments
mkdir -p experiments

# Create a log file
log_file="experiments/experiment_log_retrain.txt"
exec > >(tee -a "$log_file") 2>&1

echo "$(date): Starting experiments"

# Unlearning experiments
# Unlearn and retrain
run_experiment --model resnet --dataset cifar10 --num_clients 10 --global_rounds 20 --alpha 0.1 --batch_size 32 --local_epochs 2 --learning_rate 0.01 --unlearn --retrain --removed_clients "0,1,2"
run_experiment --model resnet --dataset cifar10 --num_clients 10 --global_rounds 20 --alpha 0.2 --batch_size 32 --local_epochs 2 --learning_rate 0.01 --unlearn --retrain --removed_clients "0,1,2"
run_experiment --model resnet --dataset cifar10 --num_clients 10 --global_rounds 20 --alpha 0.3 --batch_size 32 --local_epochs 2 --learning_rate 0.01 --unlearn --retrain --removed_clients "0,1,2"

# Add more experiments as needed

echo "$(date): All experiments completed. Results are saved in the 'experiments' directory."