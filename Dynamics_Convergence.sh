#!/bin/bash

# Array of n values
nvals=(25 50 75 100 125 150 175 200 225 250 275 300)

# Array of seed values
seeds=(0 1 2 3 4 5 6 7 8 9)

# Graphon parameters
weighted=false
graphon="hexaflake"
graphon_parameter=8

# Training parameters
dynamics="heat_sin"
num_train_trajectories=50
fourier_degree=10

# Array of available GPUs
gpus=(0 1 2 3)

# Function to create directory if it doesn't exist
create_dir_if_not_exists() {
    if [ ! -d "$1" ]; then
        echo "$1 directory does not exist. Creating it now."
        mkdir -p "$1"
    else
        echo "$1 directory already exists."
    fi
}

# Function to run each Python script
run_script() {
    local nval=$1
    local seed=$2
    local gpu=$3
    local log_file="outputs/logs/training_log_nval${nval}_seed${seed}_${graphon}${graphon_parameter}.log"
   
    # Build the weighted flag conditionally
    if [ "$weighted" = true ]; then
        weighted_flag="--use_weighted"
    else
        weighted_flag=""
    fi

    echo "Starting job with nval=$nval, seed=$seed, graphon=$graphon on GPU $gpu"
    CUDA_VISIBLE_DEVICES=$gpu python dynamics_convergence_experiment.py \
        --dynamics $dynamics \
        --nval $nval \
        $weighted_flag \
        --graphon $graphon \
        --graphon_parameter $graphon_parameter \
        --seed $seed \
        --num_train_trajectories $num_train_trajectories \
        --fourier_degree $fourier_degree > $log_file 2>&1 &
    
    echo "Job started with PID $! for nval=$nval, seed=$seed on GPU $gpu"
}

# Create logs and checkpoints directories if needed
create_dir_if_not_exists "outputs/logs"
create_dir_if_not_exists "checkpoints"

echo "Starting batch experiment with:"
echo "N values: ${nvals[*]}"
echo "Seeds: ${seeds[*]}"
echo "Graphon: $graphon (parameter: $graphon_parameter)"
echo "Training pairs per size: $num_train_trajectories"
echo "Fourier degree: $fourier_degree"
echo "GPUs: ${gpus[*]}"
echo ""

# Create all combinations of nval and seed
experiments=()
for seed in "${seeds[@]}"; do
    for nval in "${nvals[@]}"; do
        experiments+=("$nval,$seed")
    done
done

echo "Total experiments to run: ${#experiments[@]}"
echo ""

# Run experiments in batches that assign one job per GPU to avoid overloading
batch_size=${#gpus[@]}
total_experiments=${#experiments[@]}
completed=0

for ((i=0; i<total_experiments; i+=batch_size)); do
    echo "Starting batch $((i/batch_size + 1))"
    
    # Start up jobs in parallel
    for ((j=0; j<batch_size && i+j<total_experiments; j++)); do
        experiment=${experiments[$((i+j))]}
        nval=$(echo $experiment | cut -d',' -f1)
        seed=$(echo $experiment | cut -d',' -f2)
        gpu=${gpus[$j]}
        
        run_script $nval $seed $gpu
    done
    
    echo "Waiting for batch to complete."
    wait
    
    completed=$((completed + batch_size))
    if [ $completed -gt $total_experiments ]; then
        completed=$total_experiments
    fi
    
    echo "Batch completed, $completed/$total_experiments experiments done."
    echo ""
done

echo "All batch experiments completed."