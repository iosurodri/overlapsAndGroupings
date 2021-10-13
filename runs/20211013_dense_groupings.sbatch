#!/bin/bash
#SBATCH --partition=power_std
#SBATCH --account=acc_ure_power_std
#SBATCH --gres=gpu:v100:1
#SBATCH --array=1-4
# Activate the conda environment named "pytorch"
source ~/anaconda3/etc/profile.d/conda.sh
conda activate pytorch
# Move to the src/runs/ folder
cd ../src/runs
# Set all the tasks to perform
argumentos[${#argumentos[@]}]="dense --dataset CIFAR10 --pool_type grouping_product --config_file_name", "densenet_parameters.json --name dense_pool_grouping_product"
argumentos[${#argumentos[@]}]="dense --dataset CIFAR10 --pool_type grouping_maximum --config_file_name", "densenet_parameters.json --name dense_pool_grouping_maximum"
argumentos[${#argumentos[@]}]="dense --dataset CIFAR10 --pool_type grouping_ob --config_file_name", "densenet_parameters.json --name dense_pool_grouping_ob"
argumentos[${#argumentos[@]}]="dense --dataset CIFAR10 --pool_type grouping_geometric --config_file_name", "densenet_parameters.json --name dense_pool_grouping_geometric"

srun python run_test.py ${argumentos[SLURM_ARRAY_TASK_ID-1]}
