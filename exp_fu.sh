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
    local removed_clients=${11}
    local unlearn_strategy=${12}
    local penalty=${13}
    local log_file=${14}

    echo "$(date): Starting experiment: Model=$model, Dataset=$dataset, Clients=$num_clients, Alpha=$alpha, Unlearn=$unlearn, Continuous=$continuous, Removed Clients=$removed_clients, Unlearn Strategy=$unlearn_strategy, Penalty=$penalty" >> "$log_file"
    
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
        --removed_clients $removed_clients \
        --unlearn_strategy $unlearn_strategy \
        --penalty $penalty >> "$log_file" 2>&1
    
    if [ $? -eq 0 ]; then
        echo "$(date): Experiment completed successfully" >> "$log_file"
    else
        echo "$(date): Experiment failed with error code $?" >> "$log_file"
    fi
    echo "---------------------" >> "$log_file"
}

# Define experiments
experiments=(
    # "bert ag_news 10 0.2 10 32 2 2e-5 --unlearn --continuous 0,1,2 stability 0.1"
    # "bert ag_news 10 0.2 10 32 2 2e-5 --unlearn --continuous 0,1,2 stability 0.3"
    # "bert ag_news 10 0.2 10 32 2 2e-5 --unlearn --continuous 0,1,2 stability 0.5"
    # "bert ag_news 10 0.5 10 32 2 2e-5 --unlearn --continuous 0,1,2 stability 0.1"
    # "bert ag_news 10 0.5 10 32 2 2e-5 --unlearn --continuous 0,1,2 stability 0.3"
    # "bert ag_news 10 0.5 10 32 2 2e-5 --unlearn --continuous 0,1,2 stability 0.5"
    "bert ag_news 10 0.8 10 32 2 2e-5 --unlearn --continuous 0,1,2 stability 0.1"
    "bert ag_news 10 0.8 10 32 2 2e-5 --unlearn --continuous 0,1,2 stability 0.3"
    "bert ag_news 10 0.8 10 32 2 2e-5 --unlearn --continuous 0,1,2 stability 0.5"
)

main_log_file="logs/experiment_log_continuous.txt"
echo "$(date): Starting experiments" | tee -a "$main_log_file"

# Run experiments in sequence
for exp in "${experiments[@]}"; do
    log_file="logs/exp_continuous_$(date +%Y%m%d_%H%M%S).log"
    echo "$(date): Starting experiment: $exp" | tee -a "$main_log_file"
    run_experiment $exp "$log_file"
    echo "$(date): Finished experiment: $exp" | tee -a "$main_log_file"
    echo "---------------------" | tee -a "$main_log_file"
    sleep 1  # Small delay to ensure unique log file names for the next experiment
done

echo "$(date): All experiments completed. Results are saved in the 'logs' directory." | tee -a "$main_log_file"