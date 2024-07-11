
run_fu_game() {
    dataset=$1
    num_clients=$2
    alpha=$3

    echo "$(date): Starting FU game: Dataset=$dataset, Clients=$num_clients, Alpha=$alpha"
    
    python src/game.py \
        --dataset $dataset \
        --num_clients $num_clients \
        --alpha $alpha \
    
    if [ $? -eq 0 ]; then
        echo "$(date): FU game completed successfully"
    else
        echo "$(date): FU game failed with error code $?"
    fi
    echo "---------------------"

}



# FU game for the same experiments
run_fu_game mnist 10 0.5 
run_fu_game cifar10 10 0.5 

run_fu_game mnist 10 0.2
run_fu_game cifar10 10 0.2 

run_fu_game mnist 10 0.8
run_fu_game cifar10 10 0.8 

run_fu_game mnist 10 1.0
run_fu_game cifar10 10 1.0