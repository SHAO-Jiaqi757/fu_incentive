#!/bin/bash

# Function to run an experiment
run_experiment() {
    local model=$1
    local dataset=$2
    local num_clients=$3
    local alpha=$4
    local global_rounds=$5
    local batch_size=$6
    local local_epochs=$7
    local learning_rate=$8
    local unlearn=$9
    local continuous=${10}
    local unified_price=${11}
    local removed_clients=${12}
    local lambda_v=${13}
    local lambda_s=${14}
    local lambda_q=${15}
    local log_file=${16}

    echo "$(date): Starting experiment: Model=$model, Dataset=$dataset, Clients=$num_clients, Alpha=$alpha, Unlearn=$unlearn, Retrain=$continuous, Removed Clients=$removed_clients, Lambda_v=$lambda_v, Lambda_s=$lambda_s, Lambda_q=$lambda_q" >> "$log_file"
    
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
        $continuous \
        $unified_price \
        --removed_clients $removed_clients \
        --lambda_v $lambda_v \
        --lambda_s $lambda_s \
        --lambda_q $lambda_q >> "$log_file" 2>&1
    
    if [ $? -eq 0 ]; then
        echo "$(date): Experiment completed successfully" >> "$log_file"
    else
        echo "$(date): Experiment failed with error code $?" >> "$log_file"
    fi
    echo "---------------------" >> "$log_file"
}

# Function to wait for a job slot
wait_for_job_slot() {
    while [ $(jobs -r | wc -l) -ge 2 ]; do
        sleep 5
    done
}

# Create a directory for all experiments
mkdir -p experiments

# Create a main log file
main_log_file="logs/experiment_log_continuous.txt"
echo "$(date): Starting experiments" | tee -a "$main_log_file"

# Define experiments
experiments=(
    # "resnet cifar10 10 0.5 10 32 2 0.01 --unlearn --continuous 0,1,2"
    # "resnet cifar10 10 0.2 10 32 2 0.01 --unlearn --continuous 0,1,2"
    # "resnet cifar10 10 0.8 10 32 2 0.01 --unlearn --continuous 0,1,2"
    # "resnet cifar10 10 1.0 10 32 2 0.01 --unlearn --continuous 0,1,2"
    "resnet cifar100 10 0.5 100 32 2 0.01 --unlearn --continuous --unified_price 0,1,2"
    "resnet cifar100 10 0.2 100 32 2 0.01 --unlearn --continuous --unified_price 0,1,2"
    "resnet cifar100 10 0.8 100 32 2 0.01 --unlearn --continuous --unified_price 0,1,2"
    "resnet cifar100 10 1.0 100 32 2 0.01 --unlearn --continuous --unified_price 0,1,2"
    # "bert ag_news 10 0.5 10 32 2 2e-5 --unlearn --continuous --unified_price 0,1,2"
    # "bert ag_news 10 0.2 10 32 2 2e-5 --unlearn --continuous --unified_price 0,1,2"
    # "bert ag_news 10 0.8 10 32 2 2e-5 --unlearn --continuous --unified_price 0,1,2"
    # "bert ag_news 10 1.0 10 32 2 2e-5 --unlearn --continuous --unified_price 0,1,2"
)

# Define lambda sets
lambda_sets=(
    "1 1 1"
    "1000 1 1"
    "1 1000 1"
    "1 1 1000"
    "100 1 1"
    "1 100 1"
    "1 1 100"
)

# Run experiments in parallel for each lambda set
for exp in "${experiments[@]}"; do
    for lambda_set in "${lambda_sets[@]}"; do
        wait_for_job_slot
        log_file="logs/exp_continuous_$(date +%Y%m%d_%H%M%S).log"
        run_experiment $exp $lambda_set "$log_file" &
        echo "Started experiment: $exp with Lambda set: $lambda_set" | tee -a "$main_log_file"
        sleep 1  # Small delay to ensure unique log file names
    done
done

# Wait for all background jobs to finish
wait

echo "$(date): All experiments completed. Results are saved in the 'experiments' directory." | tee -a "$main_log_file"
