#!/bin/bash

run_fl() {
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
run_fl cnn mnist 10 0.5 5 32 2 0.01 64
run_fl mlp mnist 10 0.5 5 32 2 0.01 64
run_fl resnet cifar10 10 0.5 10 32 2 0.01 64

# # Varying heterogeneity (alpha)

run_fl cnn mnist 10 0.2 5 32 2 0.01 64
run_fl mlp mnist 10 0.2 5 32 2 0.01 64
run_fl resnet cifar10 10 0.2 10 32 2 0.01 64

run_fl cnn mnist 10 0.8 5 32 2 0.01 64
run_fl mlp mnist 10 0.8 5 32 2 0.01 64
run_fl resnet cifar10 10 0.8 10 32 2 0.01 64


run_fl cnn mnist 10 1.0 5 32 2 0.01 64
run_fl mlp mnist 10 1.0 5 32 2 0.01 64
run_fl resnet cifar10 10 1.0 10 32 2 0.01 64




# Add more configurations as needed

echo "$(date): All experiments completed. Results are saved in the 'experiments' directory."