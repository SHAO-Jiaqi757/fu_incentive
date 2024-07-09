#!/bin/bash

# Function to run an experiment
run_experiment() {
    model=$1
    dataset=$2
    num_clients=$3
    alpha=$4
    global_rounds=$5
    batch_size=$6
    local_epochs=$7
    learning_rate=$8
    hidden_dim=$9

    echo "$(date): Starting experiment: Model=$model, Dataset=$dataset, Clients=$num_clients, Alpha=$alpha"
    
    python federated_learning.py \
        --model $model \
        --dataset $dataset \
        --num_clients $num_clients \
        --global_rounds $global_rounds \
        --alpha $alpha \
        --batch_size $batch_size \
        --local_epochs $local_epochs \
        --learning_rate $learning_rate \
        --hidden_dim $hidden_dim
    
    if [ $? -eq 0 ]; then
        echo "$(date): Experiment completed successfully"
    else
        echo "$(date): Experiment failed with error code $?"
    fi
    echo "---------------------"

    # Wait for 30 seconds to allow GPU memory to clear
    echo "Waiting for 30 seconds before the next experiment..."
    sleep 30
}

# Create a directory for all experiments
mkdir -p experiments

# Create a log file
log_file="experiments/experiment_log.txt"
exec > >(tee -a "$log_file") 2>&1

echo "$(date): Starting experiments"

# Experiment configurations

# # Varying models and datasets
# run_experiment cnn mnist 10 0.5 5 32 2 0.01 64
# run_experiment mlp mnist 10 0.5 5 32 2 0.01 64
run_experiment resnet cifar10 10 0.5 5 32 2 0.01 64

# # Varying heterogeneity (alpha)
# run_experiment cnn mnist 10 0.1 5 32 2 0.01 64  # More heterogeneous
# run_experiment cnn mnist 10 1.0 5 32 2 0.01 64  # Less heterogeneous

# # Varying number of clients
# run_experiment mlp mnist 5 0.5 5 32 2 0.01 64
# run_experiment mlp mnist 20 0.5 5 32 2 0.01 64

# # Varying global rounds
run_experiment resnet cifar10 10 0.5 10 32 2 0.01 64

# # Varying local epochs
# run_experiment cnn mnist 10 0.5 5 32 5 0.01 64

# # Varying batch size
# run_experiment mlp mnist 10 0.5 5 64 2 0.01 64

# # Varying learning rate
run_experiment resnet cifar10 10 0.5 5 32 2 0.001 64

# Add more configurations as needed

echo "$(date): All experiments completed. Results are saved in the 'experiments' directory."