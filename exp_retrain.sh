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
    unlearn=$9
    retrain=${10}
    removed_clients=${11}

    echo "$(date): Starting experiment: Model=$model, Dataset=$dataset, Clients=$num_clients, Alpha=$alpha, Unlearn=$unlearn, Retrain=$retrain, Removed Clients=$removed_clients"
    
    python federated_learning.py \
        --model $model \
        --dataset $dataset \
        --num_clients $num_clients \
        --alpha $alpha \
        --global_rounds $global_rounds \
        --batch_size $batch_size \
        --local_epochs $local_epochs \
        --learning_rate $learning_rate \
        $unlearn \
        $retrain \
        --removed_clients $removed_clients
    
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
run_experiment cnn mnist 10 0.5 10 32 2 0.01 --unlearn --retrain "0,1,2"
run_experiment mlp mnist 10 0.5 10 32 2 0.01 --unlearn --retrain "0,1,2"
run_experiment resnet cifar10 10 0.5 20 32 2 0.01 --unlearn --retrain "0,1,2"

# # Varying heterogeneity (alpha)
run_experiment cnn mnist 10 0.2 10 32 2 0.01 --unlearn --retrain "0,1,2"
run_experiment mlp mnist 10 0.2 10 32 2 0.01 --unlearn --retrain "0,1,2"
run_experiment resnet cifar10 10 0.2 20 32 2 0.01 --unlearn --retrain "0,1,2"

run_experiment cnn mnist 10 0.8 10 32 2 0.01 --unlearn --retrain "0,1,2"
run_experiment mlp mnist 10 0.8 10 32 2 0.01 --unlearn --retrain "0,1,2"
run_experiment resnet cifar10 10 0.8 20 32 2 0.01 --unlearn --retrain "0,1,2"


run_experiment cnn mnist 10 1.0 10 32 2 0.01 --unlearn --retrain "0,1,2"
run_experiment mlp mnist 10 1.0 10 32 2 0.01 --unlearn --retrain "0,1,2"
run_experiment resnet cifar10 10 1.0 20 32 2 0.01 --unlearn --retrain "0,1,2"



# Add more experiments as needed

echo "$(date): All experiments completed. Results are saved in the 'experiments' directory."
