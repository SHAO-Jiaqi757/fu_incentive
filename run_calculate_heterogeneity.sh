#!/bin/bash

# Function to run the distance calculation
run_hetero_calculation() {
    dataset=$1
    num_clients=$2
    alpha=$3
    removed_clients=$4

    echo "$(date): Starting distance calculation: Dataset=$dataset, Clients=$num_clients, Alpha=$alpha, Removed Clients=$removed_clients"
    
    python src/evaluate_hetergeneity.py \
        --dataset $dataset \
        --num_clients $num_clients \
        --alpha $alpha \
        --removed_clients $removed_clients
    
    if [ $? -eq 0 ]; then
        echo "$(date): Distance calculation completed successfully"
    else
        echo "$(date): Distance calculation failed with error code $?"
    fi
    echo "---------------------"

}



# Create a directory for all experiments
mkdir -p experiments


echo "$(date): Starting experiments"

# Distance calculations for the same experiments
run_hetero_calculation mnist 10 0.5 "0,1,2"
run_hetero_calculation cifar10 10 0.5 "0,1,2"


run_hetero_calculation mnist 10 0.2 "0,1,2"
run_hetero_calculation cifar10 10 0.2 "0,1,2"


run_hetero_calculation mnist 10 0.8 "0,1,2"
run_hetero_calculation cifar10 10 0.8 "0,1,2"


run_hetero_calculation mnist 10 1.0 "0,1,2"
run_hetero_calculation cifar10 10 1.0 "0,1,2"

