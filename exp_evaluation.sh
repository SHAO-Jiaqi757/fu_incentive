#!/bin/bash

# Function to run an experiment
run_evaluation() {
    local model=$1
    local dataset=$2
    local num_clients=$3
    local alpha=$4
    local unlearn=$5
    local continuous=$6
    local retrain=$7
    local lambda_v=$8
    local lambda_s=$9
    local lambda_q=${10}
    local unified_price=${11}
    local log_file=${12}

    echo "$(date): Starting experiment: Model=$model, Dataset=$dataset, Clients=$num_clients, Alpha=$alpha, Unlearn=$unlearn, Retrain=$retrain, Continuous=$continuous, Lambda_v=$lambda_v, Lambda_s=$lambda_s, Lambda_q=$lambda_q" >> "$log_file"
    
    python evaluation.py \
        --model $model \
        --dataset $dataset \
        --num_clients $num_clients \
        --alpha $alpha \
        $unlearn \
        $continuous \
        $retrain \
        $unified_price \
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

run_analysis() {
    local model=$1
    local dataset=$2
    local num_clients=$3
    local alpha=$4
    local lambda_v=$5
    local lambda_s=$6
    local lambda_q=$7
    local unified_price=${8}
    local log_file=${9}

    echo "$(date): Starting analysis experiment: Model=$model, Dataset=$dataset, Clients=$num_clients, Alpha=$alpha, Lambda_v=$lambda_v, Lambda_s=$lambda_s, Lambda_q=$lambda_q" >> "$log_file"
    
    python analyze.py \
        --model $model \
        --dataset $dataset \
        --num_clients $num_clients \
        --alpha $alpha \
        --lambda_v $lambda_v \
        --lambda_s $lambda_s \
        --lambda_q $lambda_q \
        $unified_price >> "$log_file" 2>&1
    
    if [ $? -eq 0 ]; then
        echo "$(date): Experiment completed successfully" >> "$log_file"
    else
        echo "$(date): Experiment failed with error code $?" >> "$log_file"
    fi
    echo "---------------------" >> "$log_file"
}

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

# Function to wait for a job slot
wait_for_job_slot() {
    while [ $(jobs -r | wc -l) -ge 4 ]; do
        sleep 5
    done
}

# Define experiments for evaluation
evaluation_experiments_continuous=(
    # "resnet cifar10 10 0.2 --unlearn --continuous"
    # "resnet cifar10 10 0.5 --unlearn --continuous"
    # "resnet cifar10 10 0.8 --unlearn --continuous"
    # "resnet cifar10 10 1.0 --unlearn --continuous"
    # "resnet cifar100 10 0.2 --unlearn --continuous"
    # "resnet cifar100 10 0.5 --unlearn --continuous"
    # "resnet cifar100 10 0.8 --unlearn --continuous"
    # "resnet cifar100 10 1.0 --unlearn --continuous"
    "resnet cifar100 10 0.2 --unlearn --continuous --unified_price"
    "resnet cifar100 10 0.5 --unlearn --continuous --unified_price"
    "resnet cifar100 10 0.8 --unlearn --continuous --unified_price"
    "resnet cifar100 10 1.0 --unlearn --continuous --unified_price"
    # "bert ag_news 10 0.5 --unlearn --continuous"
    # "bert ag_news 10 0.2 --unlearn --continuous"
    # "bert ag_news 10 0.8 --unlearn --continuous"
    # "bert ag_news 10 1.0 --unlearn --continuous"
    # "bert ag_news 10 0.5 --unlearn --continuous --unified_price"
    # "bert ag_news 10 0.2 --unlearn --continuous --unified_price"
    # "bert ag_news 10 0.8 --unlearn --continuous --unified_price"
    # "bert ag_news 10 1.0 --unlearn --continuous --unified_price"
)

evaluation_experiments_retrain=(
    # "resnet cifar10 10 0.2 --unlearn --retrain"
    # "resnet cifar10 10 0.5 --unlearn --retrain"
    # "resnet cifar10 10 0.8 --unlearn --retrain"
    # "resnet cifar10 10 1.0 --unlearn --retrain"
    # "resnet cifar100 10 0.2 --unlearn --retrain"
    # "resnet cifar100 10 0.5 --unlearn --retrain"
    # "resnet cifar100 10 0.8 --unlearn --retrain"
    # "resnet cifar100 10 1.0 --unlearn --retrain"
    "bert ag_news 10 0.5 --unlearn --retrain"
    "bert ag_news 10 0.2 --unlearn --retrain"
    "bert ag_news 10 0.8 --unlearn --retrain"
    "bert ag_news 10 1.0 --unlearn --retrain"
)

# Define experiments for analysis
analysis_experiments=(
    "resnet cifar100 10 0.5"
    "resnet cifar100 10 0.2"
    "resnet cifar100 10 0.8"
    "resnet cifar100 10 1.0"
    "resnet cifar100 10 0.5 --unified_price"
    "resnet cifar100 10 0.2 --unified_price"
    "resnet cifar100 10 0.8 --unified_price"
    "resnet cifar100 10 1.0 --unified_price"
    # "resnet cifar10 10 0.5"
    # "resnet cifar10 10 0.2"
    # "resnet cifar10 10 0.8"
    # "resnet cifar10 10 1.0"
    # "bert ag_news 10 0.5 --unified_price"
    # "bert ag_news 10 0.2 --unified_price"
    # "bert ag_news 10 0.8 --unified_price"
    # "bert ag_news 10 1.0 --unified_price"
)

# Create a directory for all experiments
mkdir -p logs 

# Create a main log file
main_log_file="logs/experiment_log.txt"
echo "$(date): Starting experiments" | tee -a "$main_log_file"

# Run evaluation experiments with each lambda set
# for exp in "${evaluation_experiments_continuous[@]}"; do
#     for lambda_set in "${lambda_sets[@]}"; do
#         wait_for_job_slot
#         log_file="logs/eval_continuous_$(date +%Y%m%d_%H%M%S).log"
#         read -r model dataset num_clients alpha unlearn continuous unified_price <<< "$exp"
#         read -r lambda_v lambda_s lambda_q <<< "$lambda_set"
#         run_evaluation "$model" "$dataset" "$num_clients" "$alpha" "$unlearn" "$continuous" "" "$lambda_v" "$lambda_s" "$lambda_q" "$unified_price" "$log_file" &
#         echo "Started experiment: $exp $lambda_set" | tee -a "$main_log_file"
#         sleep 1
#     done
# done

# wait
# for exp in "${evaluation_experiments_retrain[@]}"; do
#     wait_for_job_slot
#     log_file="logs/eval_retrain_$(date +%Y%m%d_%H%M%S).log"
#     read -r model dataset num_clients alpha unlearn continuous unified_price <<< "$exp"
#     run_evaluation "$model" "$dataset" "$num_clients" "$alpha" "$unlearn" "" "--retrain" "" "" "" "$unified_price" "$log_file" &
#     echo "Started experiment: $exp" | tee -a "$main_log_file"
#     sleep 1
# done
# Run analysis experiments with each lambda set
for exp in "${analysis_experiments[@]}"; do
    for lambda_set in "${lambda_sets[@]}"; do
        read -r model dataset num_clients alpha unified_price <<< "$exp"
        read -r lambda_v lambda_s lambda_q <<< "$lambda_set"
        log_file="logs/analysis_${model}_${dataset}_${num_clients}_${alpha}_${lambda_set}.log"
        run_analysis "$model" "$dataset" "$num_clients" "$alpha" "$lambda_v" "$lambda_s" "$lambda_q" "$unified_price" "$log_file"
        echo "Started analysis: $exp $lambda_set" | tee -a "$main_log_file"
        sleep 1
    done
done

# Wait for all background jobs to finish
wait

echo "$(date): All experiments completed. Results are saved in the 'experiments' directory." | tee -a "$main_log_file"