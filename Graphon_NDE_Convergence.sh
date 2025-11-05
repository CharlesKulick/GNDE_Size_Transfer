#!/bin/bash

# Bash script to run all experiments sequentially
echo "Starting convergence experiments."

echo "Running tent graphon computation."
nohup python convergence_experiment.py --graphon tent --graphon_parameter 0 --num_random_inits 100 > experiment_log_tent.log 2>&1
echo "Tent graphon computation completed."

echo "Running HSBM graphon computation."
nohup python convergence_experiment.py --graphon hsbm --graphon_parameter 5 --num_random_inits 100 > experiment_log_hsbm.log 2>&1
echo "HSBM graphon computation completed."

echo "Running hexaflake graphon computation."
nohup python convergence_experiment.py --graphon hexaflake --graphon_parameter 10 --num_random_inits 100 > experiment_log_hexaflake.log 2>&1
echo "Hexaflake graphon computation completed."

echo "All experiments completed!"