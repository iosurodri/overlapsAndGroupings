import os
import sys
sys.path.append(os.path.abspath(os.path.join('..', '..')))

# Data loading and saving:
from src.data.load_data import load_dataset

# Model interaction:
from src.model_tools.evaluate import get_prediction_metrics
from src.model_tools.load_model import load_model

# Auxiliar modules
import torch


def run_evaluation(model_file_name, model_type='lenet', info_file_name=None, dataset='CIFAR10'):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = 64
    test_loader = load_dataset(dataset, batch_size=batch_size, train=False, num_workers=0, pin_memory=True)
    model = load_model(model_file_name, model_type=model_type, info_file_name=info_file_name).to(device)
    prediction_metrics = get_prediction_metrics(model, device=device, test_loader=test_loader)
    print(prediction_metrics)


if __name__ == '__main__':
    # model_file_name = 'COVID4_2layerclassifier__lr0001__partitioned__adam_checkpoint'
    base_folder = ''
    model_file_name = 'lenet_product'
    model_file_name = os.path.join(base_folder, model_file_name)
    model_type = 'lenet'
    info_file_name = model_file_name + '_info.json'
    info_file_name = os.path.join(base_folder, info_file_name)
    dataset = 'CIFAR10'
    run_evaluation(model_file_name, model_type=model_type, info_file_name=info_file_name, dataset=dataset)
