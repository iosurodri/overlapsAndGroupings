import os
from collections import Mapping

PATH_ROOT = os.path.join('..', '..', 'reports')


def log_eval_results(name, acc, loss=None):
    results_folder = os.path.join(PATH_ROOT, 'results')
    try:
        os.mkdir(results_folder)
    except:
        pass
    results_folder = os.path.join(results_folder, name)
    try:
        os.mkdir(results_folder)
    except:
        pass
    with open(os.path.join(results_folder, 'test_eval.txt'), 'w') as f:
        f.write('{}\n'.format(acc))
        if loss is not None:
            f.write('{}'.format(loss))


def log_eval_metrics(name, metrics):
    # Start by checking if metrics is a dictionary of metrics and values:
    if not isinstance(metrics, Mapping):
        raise Exception('Metrics is expected to be a dictionary of the form "metric_name" -> "metric value"')
    results_folder = os.path.join(PATH_ROOT, 'results')
    try:
        os.mkdir(results_folder)
    except:
        pass
    results_folder = os.path.join(results_folder, name)
    try:
        os.mkdir(results_folder)
    except:
        pass
    with open(os.path.join(results_folder, 'test_metrics.txt'), 'w') as f:
        for metric_name in metrics.keys():
            f.write('{metric_name}: {metric_value}\n'.format(metric_name=metric_name, metric_value=metrics[metric_name]))
