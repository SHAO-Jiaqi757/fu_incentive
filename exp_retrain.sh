#!/bin/bash

run_retrain() {
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
        --model "$model" \
        --dataset "$dataset" \
        --num_clients "$num_clients" \
        --alpha "$alpha" \
        --global_rounds "$global_rounds" \
        --batch_size "$batch_size" \
        --local_epochs "$local_epochs" \
        --learning_rate "$learning_rate" \
        "$unlearn" \
        "$retrain" \
        --removed_clients $removed_clients

    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "$(date): Experiment completed successfully" | tee -a "$log_file"
    else
        echo "$(date): Experiment failed with error code $exit_code" | tee -a "$log_file"
    fi
    echo "---------------------" | tee -a "$log_file"
    echo "---------------------" >> "$log_file"
}

# Create a directory for all experiments
mkdir -p experiments

# Create a log file
main_log_file="logs/experiment_retrain_log.txt"

echo "$(date): Starting experiments" | tee -a "$main_log_file"

# Define experiment configurations
experiments=(
    "resnet cifar100 10 0.5 100 32 2 0.01 --unlearn --retrain 0,1,2"
    # "bert ag_news 10 0.5 50 32 2 2e-5 --unlearn --retrain 0,1,2"
    "resnet cifar100 10 0.2 100 32 2 0.01 --unlearn --retrain 0,1,2"
    # "bert ag_news 10 0.2 50 32 2 2e-5 --unlearn --retrain 0,1,2"
    "resnet cifar100 10 0.8 100 32 2 0.01 --unlearn --retrain 0,1,2"
    # "bert ag_news 10 0.8 50 32 2 2e-5 --unlearn --retrain 0,1,2"
    "resnet cifar100 10 1.0 100 32 2 0.01 --unlearn --retrain 0,1,2"
    # "bert ag_news 10 1.0 50 32 2 2e-5 --unlearn --retrain 0,1,2"
)

# Function to wait for a job slot
wait_for_job_slot() {
    while [ $(jobs -r | wc -l) -ge 2 ]; do
        sleep 5
    done
}

# Run experiments
for exp in "${experiments[@]}"; do
    wait_for_job_slot
    log_file="logs/exp_retrain_$(date +%Y%m%d_%H%M%S).log"
    run_retrain $exp "$log_file" &
    echo "Started experiment: $exp" | tee -a "$main_log_file"
    sleep 30  # Wait 30 seconds before starting the next experiment
done

# Wait for all background jobs to finish
wait

echo "$(date): All experiments completed. Results are saved in the 'experiments' directory." | tee -a "$main_log_file"