#!/bin/bash

# Function to run an experiment
run_evaluation() {
    model=$1
    dataset=$2
    num_clients=$3
    alpha=$4
    unlearn=$5
    continuous=$6
    retrain=$7

    echo "$(date): Starting experiment: Model=$model, Dataset=$dataset, Clients=$num_clients, Alpha=$alpha, Unlearn=$unlearn, Retrain=$retrain, Continuous=$continuous"
    
    python evaluation.py \
        --model $model \
        --dataset $dataset \
        --num_clients $num_clients \
        --alpha $alpha \
        $unlearn \
        $continuous \
        $retrain
    
    if [ $? -eq 0 ]; then
        echo "$(date): Experiment completed successfully"
    else
        echo "$(date): Experiment failed with error code $?"
    fi
    echo "---------------------"

    echo "Waiting for 30 seconds before the next experiment..."
    sleep 30
}

run_analysis() {
    model=$1
    dataset=$2
    num_clients=$3
    alpha=$4

    echo "$(date): Starting analysis experiment: Model=$model, Dataset=$dataset, Clients=$num_clients, Alpha=$alpha"
    
    python analyze.py \
        --model $model \
        --dataset $dataset \
        --num_clients $num_clients \
        --alpha $alpha \
    
    if [ $? -eq 0 ]; then
        echo "$(date): Experiment completed successfully"
    else
        echo "$(date): Experiment failed with error code $?"
    fi
    echo "---------------------"
}
# Create a directory for all experiments
mkdir -p experiments

# # Create a log file
# log_file="experiments/experiment_evaluation.txt"
# exec > >(tee -a "$log_file") 2>&1

# echo "$(date): Starting experiments"

# # Unlearning experiments
# run_evaluation cnn mnist 10 0.5  --unlearn --continuous 
# run_evaluation mlp mnist 10 0.5  --unlearn --continuous 
# run_evaluation resnet cifar10 10 0.5  --unlearn --continuous 

# # # Varying heterogeneity (alpha)
# run_evaluation cnn mnist 10 0.2  --unlearn --continuous 
# run_evaluation mlp mnist 10 0.2  --unlearn --continuous 
# run_evaluation resnet cifar10 10 0.2  --unlearn --continuous 

# run_evaluation cnn mnist 10 0.8  --unlearn --continuous 
# run_evaluation mlp mnist 10 0.8  --unlearn --continuous 
# run_evaluation resnet cifar10 10 0.8  --unlearn --continuous 


# run_evaluation cnn mnist 10 1.0  --unlearn --continuous 
# run_evaluation mlp mnist 10 1.0  --unlearn --continuous 
# run_evaluation resnet cifar10 10 1.0  --unlearn --continuous 

# # Retraining experiments
# run_evaluation cnn mnist 10 0.5  --unlearn --retrain
# run_evaluation mlp mnist 10 0.5  --unlearn --retrain 
# run_evaluation resnet cifar10 10 0.5  --unlearn --retrain 

# # # Varying heterogeneity (alpha)
# run_evaluation cnn mnist 10 0.2  --unlearn --retrain 
# run_evaluation mlp mnist 10 0.2  --unlearn --retrain 
# run_evaluation resnet cifar10 10 0.2  --unlearn --retrain 

# run_evaluation cnn mnist 10 0.8  --unlearn --retrain 
# run_evaluation mlp mnist 10 0.8  --unlearn --retrain 
# run_evaluation resnet cifar10 10 0.8  --unlearn --retrain 


# run_evaluation cnn mnist 10 1.0  --unlearn --retrain 
# run_evaluation mlp mnist 10 1.0  --unlearn --retrain 
# run_evaluation resnet cifar10 10 1.0  --unlearn --retrain 


# Create a log file
log_file="experiments/experiment_analysis.txt"
exec > >(tee -a "$log_file") 2>&1

echo "$(date): Starting experiments"

# Unlearning experiments
run_analysis cnn mnist 10 0.5 
run_analysis mlp mnist 10 0.5 
run_analysis resnet cifar10 10 0.5 

# # Varying heterogeneity (alpha)
run_analysis cnn mnist 10 0.2 
run_analysis mlp mnist 10 0.2 
run_analysis resnet cifar10 10 0.2 

run_analysis cnn mnist 10 0.8 
run_analysis mlp mnist 10 0.8 
run_analysis resnet cifar10 10 0.8 


run_analysis cnn mnist 10 1.0 
run_analysis mlp mnist 10 1.0 
run_analysis resnet cifar10 10 1.0 

# Retraining experiments
run_analysis cnn mnist 10 0.5
run_analysis mlp mnist 10 0.5 
run_analysis resnet cifar10 10 0.5 

# # Varying heterogeneity (alpha)
run_analysis cnn mnist 10 0.2 
run_analysis mlp mnist 10 0.2 
run_analysis resnet cifar10 10 0.2 

run_analysis cnn mnist 10 0.8 
run_analysis mlp mnist 10 0.8 
run_analysis resnet cifar10 10 0.8 


run_analysis cnn mnist 10 1.0 
run_analysis mlp mnist 10 1.0 
run_analysis resnet cifar10 10 1.0 




# Add more experiments as needed

echo "$(date): All experiments completed. Results are saved in the 'experiments' directory."
