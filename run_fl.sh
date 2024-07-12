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
    log_file=$10
    echo "$(date): Starting experiment: Model=$model, Dataset=$dataset, Clients=$num_clients, Alpha=$alpha" >> "$log_file"
    
    python federated_learning.py \
        --model $model \
        --dataset $dataset \
        --num_clients $num_clients \
        --global_rounds $global_rounds \
        --alpha $alpha \
        --batch_size $batch_size \
        --local_epochs $local_epochs \
        --learning_rate $learning_rate \
        --hidden_dim $hidden_dim >> "$log_file" 2>&1
    
    exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "$(date): Experiment completed successfully" >> "$log_file"
    else
        echo "$(date): Experiment failed with error code $exit_code" >> "$log_file"
    fi
    echo "---------------------" >> "$log_file"
}

# Create a directory for all experiments
mkdir -p experiments

# Create a log file
main_log_file="experiments/experiment_log.txt"

echo "$(date): Starting experiments" | tee -a "$main_log_file"

# Define experiment configurations
experiments=(
    "resnet cifar100 10 0.5 100 32 2 0.01 64"
    "bert ag_news 10 0.5 50 32 2 2e-5 32"
    "resnet cifar100 10 0.2 100 32 2 0.01 64"
    "bert ag_news 10 0.2 50 32 2 2e-5 32"
    "resnet cifar100 10 0.8 100 32 2 0.01 64"
    "bert ag_news 10 0.8 50 32 2 2e-5 32"
    "resnet cifar100 10 1.0 100 32 2 0.01 64"
    "bert ag_news 10 1.0 50 32 2 2e-5 32"
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
    log_file="experiments/exp_$(date +%Y%m%d_%H%M%S).log"
    run_fl $exp "$log_file" &
    echo "Started experiment: $exp" | tee -a "$main_log_file"
    sleep 30  # Wait 30 seconds before starting the next experiment
done

# Wait for all background jobs to finish
wait

echo "$(date): All experiments completed. Results are saved in the 'experiments' directory." | tee -a "$main_log_file"