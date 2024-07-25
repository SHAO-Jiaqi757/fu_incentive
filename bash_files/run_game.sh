#!/bin/bash

run_fu_game() {
    dataset=$1
    num_clients=$2
    alpha=$3
    lambda_v=$4
    lambda_s=$5
    lambda_q=$6

    echo "$(date): Starting FU game: Dataset=$dataset, Clients=$num_clients, Alpha=$alpha, Lambda_v=$lambda_v, Lambda_s=$lambda_s, Lambda_q=$lambda_q"
    
    python src/unified_price.py \
        --dataset $dataset \
        --num_clients $num_clients \
        --alpha $alpha \
        --lambda_v $lambda_v \
        --lambda_s $lambda_s \
        --lambda_q $lambda_q
    
    if [ $? -eq 0 ]; then
        echo "$(date): FU game completed successfully"
    else
        echo "$(date): FU game failed with error code $?"
    fi
    echo "---------------------"
}

# FU game parameters
# dataset=(cifar100 cifar10 ag_news)
dataset=(ag_news)
num_clients=10
alphas=(0.5 0.2 0.8 1.0)
lambda_sets=(
    "1 1 1"
    "1000 1 1"
    "1 1000 1"
    "1 1 1000"
    "100 1 1"
    "1 100 1"
    "1 1 100"
)

# Run FU game for all combinations of alphas and lambda sets
for dataset in "${dataset[@]}"; do
    for alpha in "${alphas[@]}"; do
        for lambda_set in "${lambda_sets[@]}"; do
            run_fu_game $dataset $num_clients $alpha $lambda_set >> "experiments/fu_game_${dataset}_${num_clients}.txt"
        done
    done
done
