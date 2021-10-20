#!/bin/bash
#SBATCH --partition=power_std
#SBATCH --account=acc_ure_power_std
#SBATCH --gres=gpu:v100:1
#SBATCH --array=1-7
# Activate the conda environment named "pytorch"
source ~/anaconda3/etc/profile.d/conda.sh
conda activate pytorch
# Move to the src/runs/ folder
cd ../src/runs
# Set all the tasks to perform
argumentos[${#argumentos[@]}]="nin --dataset CIFAR10 --pool_type grouping_plus_product --config_file_name 100epochs_parameters.json --log_param_dist --initial_pool_exp 0.1 --name nin_pool_grouping_plus_prod_lr0_1"
argumentos[${#argumentos[@]}]="nin --dataset CIFAR10 --pool_type grouping_plus_product --config_file_name 100epochs_parameters.json --log_param_dist --initial_pool_exp 0.25 --name nin_pool_grouping_plus_prod_lr0_25"
argumentos[${#argumentos[@]}]="nin --dataset CIFAR10 --pool_type grouping_plus_product --config_file_name 100epochs_parameters.json --log_param_dist --initial_pool_exp 0.5 --name nin_pool_grouping_plus_prod_lr0_5"
argumentos[${#argumentos[@]}]="nin --dataset CIFAR10 --pool_type grouping_plus_product --config_file_name 100epochs_parameters.json --log_param_dist --initial_pool_exp 1 --name nin_pool_grouping_plus_prod_lr1"
argumentos[${#argumentos[@]}]="nin --dataset CIFAR10 --pool_type grouping_plus_product --config_file_name 100epochs_parameters.json --log_param_dist --initial_pool_exp 1.5 --name nin_pool_grouping_plus_prod_lr1_5"
argumentos[${#argumentos[@]}]="nin --dataset CIFAR10 --pool_type grouping_plus_product --config_file_name 100epochs_parameters.json --log_param_dist --initial_pool_exp 1.75 --name nin_pool_grouping_plus_prod_lr1_75"
argumentos[${#argumentos[@]}]="nin --dataset CIFAR10 --pool_type grouping_plus_product --config_file_name 100epochs_parameters.json --log_param_dist --initial_pool_exp 2 --name nin_pool_grouping_plus_prod_lr2"


srun python run_test.py ${argumentos[SLURM_ARRAY_TASK_ID-1]}
