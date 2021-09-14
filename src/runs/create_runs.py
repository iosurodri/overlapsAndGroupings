import os
import json
import time
import argparse

from itertools import combinations

CONFIG_PATH = os.path.join('..', '..', 'config', 'runs', '')
RUNS_PATH = os.path.join('..', '..', 'runs', '')
TEMPLATE_FILE = os.path.join(RUNS_PATH, 'template_run.sbatch')


def parse_args():
    # Prepare argument parser:
    CLI = argparse.ArgumentParser()
    CLI.add_argument("test_file", nargs='?', type=str, default='tests_template.json', help='File with the specification of the '
                                                                                    'jobs to perform. Saved in config/runs/')
    # CLI.add_argument("--num_jobs", nargs=1, type=int, help='Number of jobs to run in parallel (default = 1)')
    # Parse command line arguments:
    return CLI.parse_args()

#def generate_jobs(networks, datasets, pools, activations, optimizers, learning_rates, weight_decais, num_jobs=1):
def generate_jobs(networks, datasets, pools):

    current_time = time.localtime(time.time())
    jobs_id = 'datascience_job_%04d_%02d_%02d__%02d_%02d_pool' % (current_time.tm_year, current_time.tm_mon,
                                                                  current_time.tm_mday, current_time.tm_hour,
                                                                  current_time.tm_min)

    num_tasks = len(networks) * len(datasets) * len(pools)  # ToDo: Precompute:

    with open(os.path.join(RUNS_PATH, jobs_id+'.sbatch'), mode='w') as fTasks:
        # Start copying the initial lines from the template file into the new file:
        with open(TEMPLATE_FILE, mode='r') as fTemplate:
            lines = fTemplate.readlines()
            for line in lines:
                if "--array" in line:
                    fTasks.write(line[:-1])
                    # Specify the number of jobs to be performed:
                    fTasks.write("1-{}\n".format(str(num_tasks)))
                else:
                    fTasks.write(line)
        # Copy the different commands:
        base_job = 'argumentos[${#argumentos[@]}]="'
        base_name = ''
        for network in networks:
            network_job = base_job + '{} '.format(network)
            network_name = base_name + '{}_'.format(network)
            for dataset in datasets:
                dataset_job = network_job + '--dataset {} '.format(dataset)
                dataset_name = network_name + '{}_'.format(dataset)
                for pool in pools:
                    base_pool_job = dataset_job + '--pool '
                    base_pool_name = dataset_name + 'pool'
                    pool_job = base_pool_job + '{} '.format(pool[0])
                    pool_name = base_pool_name + '_{}'.format(pool[0])
                    for aggr in pool[1:]:
                        pool_job = pool_job + '{} '.format(aggr)
                        pool_name = pool_name + '_{}'.format(aggr)
                    # Print the final job:
                    final_job = pool_job + '--name {}"\n'.format(pool_name)
                    fTasks.write(final_job)
        # Write the srun command for running all generated jobs:
        fTasks.write("srun python run_model.py ${argumentos[SLURM_ARRAY_TASK_ID-1]}")

if __name__ == '__main__':
    args = parse_args()
    test_file_name = args.test_file
    with open(os.path.join(CONFIG_PATH, test_file_name)) as test_file:
        test_data = json.load(test_file)
        networks = test_data['network']
        datasets = test_data['dataset']
        pools = test_data['pool']
        # activations = test_data['activation']
        # optimizers = test_data['optimizer']
        # learning_rates = test_data['learning_rate']
        # weight_decais = test_data['weight_decay']
    generate_jobs(networks, datasets, pools)
    # generate_jobs(networks, datasets, pools, activations, optimizers, learning_rates, weight_decais, num_jobs=num_jobs)
    print('Jobs have been generated')