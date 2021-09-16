import os
import sys
sys.path.append(os.path.abspath(os.path.join('..','..')))
import argparse
import re
import torch
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from src.model_tools.load_model import load_model


PATH_MODELS = os.path.join('..', '..', 'reports', 'models', '')


def parse_args():
    # Prepare argument parser:
    CLI = argparse.ArgumentParser()
    CLI.add_argument("--folder_name", nargs="?", type=str, help="""Name of the parent folder inside of which all models
        are stored.""")
    CLI.add_argument("--model_type", nargs="?", type=str, help="""Determines the type of architecture to load. 
        'ref' for REFNet and 'Standard' for StandardNet.""")
    CLI.add_argument("--multiple_runs", nargs="?", type=bool, default=False, help="""Determines if there exist multiple
        models with same structure as result of having performed 'n' tests by model.""")
    return CLI.parse_args()


def record_predictions(model, output_file, dataset=CIFAR10, show_acc=True):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Prepare dataset:
    test_transform = transforms.Compose(
        [transforms.ToTensor()]
    )
    test_dataset = dataset(root='../../data/external', train=False,
                           download=True, transform=test_transform)
    test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)
    softmax = torch.nn.Softmax(dim=1).to(device)
    if show_acc:
        correct = 0
        total = 0
    with open(output_file, 'w') as f:
        with torch.no_grad():
            for data in test_dataloader:
                # Get the inputs; data is a list of [inputs, labels]
                inputs, labels = data[0].to(device), data[1].to(device)
                # ToDo: Hardcoded for use with network 'reference':
                outputs = model(inputs)
                outputs = softmax(outputs)
                if show_acc:
                    total += labels.shape[0]
                    correct += torch.sum(outputs.argmax(dim=1) == labels).item()
                outputs_np = outputs.cpu().numpy()
                for prediction in outputs_np:
                    f.write('{:.18e}'.format(prediction[0]))  # We use standard numpy dump to file format for floats
                    for probability in prediction[1:]:
                        f.write(', {:.18e}'.format(probability))
                    f.write('\n')
    if show_acc:
        print('Accuracy: {}'.format(correct / total))

if __name__ == '__main__':
    args = parse_args()
    folder_name = args.folder_name
    folder_name = os.path.join(PATH_MODELS, folder_name)
    model_type = args.model_type
    multiple_runs = args.multiple_runs
    files_list = os.listdir(folder_name)
    # We filter all json files, since we are interested exclusively in the models now:
    files_list_filtered = list(filter(lambda x: not x.endswith('.json'), files_list))
    files_list_filtered = list(filter(lambda x: not x.endswith('.csv'), files_list_filtered))  # Ignore previous runs of this script
    # We have several trained models with each architecture, but we'll analize just one of them:
    if multiple_runs:
        already_evaluated = []
        to_be_evaluated = []
        for file in files_list_filtered:
            not_already_evaluated = True
            for value in already_evaluated:
                if file.startswith(value):
                    not_already_evaluated = False
            if not_already_evaluated:
                # Mantemenos este modelo:
                to_be_evaluated.append(file)
                already_evaluated.append(re.search('(.*)_[0-9]', file).group(1))
    else:
        to_be_evaluated = files_list_filtered
    for file in to_be_evaluated:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        info_file = os.path.join(folder_name, file + '.json')
        if os.path.exists(info_file):
            net = load_model(os.path.join(folder_name, file), model_type, info_file)
        else:
            net = load_model(os.path.join(folder_name, file), model_type)
        net.to(device)
        net.eval()

        record_predictions(net, output_file=os.path.join(folder_name, file + '.csv'), dataset=CIFAR10)

