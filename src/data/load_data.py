import os
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np

import torch
from functools import reduce

PATH_DATA = os.path.join('..', '..', 'data', 'processed')

datasets_info = {
    'CIFAR10': {
        'dataset': datasets.CIFAR10,
        'mean': (0.485, 0.456, 0.406),
        'std': (0.229, 0.224, 0.225),
    }
}


def get_mean(dataset_train):
    num_samples = len(dataset_train)
    to_tensor = transforms.ToTensor()
    sample_shape = to_tensor(dataset_train[0][0]).shape
    num_pixels = reduce(lambda x, y: x * y, sample_shape)
    num_channels = sample_shape[0]

    total_intensity = torch.zeros(num_channels)
    # Compute global mean of the whole sample (both train and test) set:
    for img in dataset_train:
        for channel in range(num_channels):
            total_intensity[channel] += torch.sum(to_tensor(img[0])[channel, :])
    dataset_mean = total_intensity / (num_samples * num_pixels)
    return dataset_mean


def get_std(dataset_train, dataset_mean):
    num_samples = len(dataset_train)
    to_tensor = transforms.ToTensor()
    sample_shape = to_tensor(dataset_train[0][0]).shape
    num_pixels = reduce(lambda x, y: x * y, sample_shape)
    num_channels = sample_shape[0]

    # Compute global standard deviation
    total_diff = torch.zeros(num_channels)
    for img in dataset_train:
        for channel in range(num_channels):
            total_diff[channel] += torch.sum(torch.pow(to_tensor(img[0])[channel, :] - dataset_mean[channel], 2))
    dataset_std = torch.sqrt(total_diff / (num_samples * num_pixels))
    return dataset_std


def load_dataset(dataset_name, batch_size=32, train=True, train_proportion=0.8, val=True, num_workers=1, pin_memory=True):

    if dataset_name not in datasets_info.keys():
        raise Exception('No entry for dataset {}: Provided dataset must be one of {}'.format(
            dataset_name, datasets_info.keys()))

    # 1.-Prepare transformations to be applied to each set:
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(datasets_info[dataset_name]['mean'], datasets_info[dataset_name]['std'])]
    )
    # 2.-Prepare the datasets:
    if train:
        train_dataset = datasets_info[dataset_name]['dataset'](
            root=os.path.join('..', '..', 'data', 'external'), train=True,
            download=True, transform=transform,
        )
    else:
        test_dataset = datasets_info[dataset_name]['dataset'](
            root=os.path.join('..', '..', 'data', 'external'), train=False,
            download=True, transform=transform,
        )
    # 3.-Prepare the DataLoaders using the previous Datasets
    if train:
        if val:
            # Split the train dataset into train and validation sets:
            # Get a random split with a proportion of train_proportion samples for the train subset and the remaining
            # ones for the validation subset:
            num_train = len(train_dataset)
            indices = list(range(num_train))
            split = int(np.floor(train_proportion * num_train))
            np.random.shuffle(indices)
            # Generate some SubsetRandomSampler with the indexes of the images corresponding to each subset:
            train_idx, val_idx = indices[:split], indices[split:]
            train_sampler = SubsetRandomSampler(train_idx)
            val_sampler = SubsetRandomSampler(val_idx)
            # Generate DataLoader for the images:
            train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler,
                                      num_workers=num_workers,
                                      pin_memory=pin_memory)
            val_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=val_sampler, num_workers=num_workers,
                                    pin_memory=pin_memory)
        else:
            train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers,
                                      pin_memory=pin_memory, shuffle=True)
    else:
        test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
    # 4.-Return the requested DataLoaders:
    if train:
        if val:
            return train_loader, val_loader
        else:
            return train_loader
    else:
        return test_loader


if __name__ == '__main__':

    import matplotlib.pyplot as plt

    dataset_name = 'CIFAR10'
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(datasets_info[dataset_name]['mean'], datasets_info[dataset_name]['std'])]
    )
    # 2.-Prepare the datasets:
    train_dataset = datasets_info[dataset_name]['dataset'](
        root=os.path.join('..', '..', 'data', 'external'), train=True,
        download=True, transform=transform,
    )

    sample = train_dataset[np.random.randint(len(train_dataset))][0]
    sample_shape = sample.shape

    plt.imshow(sample.permute(1, 2, 0).numpy())
    plt.show()

    k = 2

    sample_unfolded = sample.unfold(1, k, k).unfold(2, k, k)
    sample_unfolded = sample_unfolded.reshape(sample_unfolded.shape[0], sample_unfolded.shape[1] * sample_unfolded.shape[2],
                                              sample_unfolded.shape[3] * sample_unfolded.shape[4])

    U, eigenvalues, eigenvectors = torch.pca_lowrank(sample_unfolded, center=True)

    sample_projected = torch.matmul(sample_unfolded, eigenvectors[..., :1])
    sample_projected = sample_projected.reshape(sample_shape[0], int(sample_shape[1] / k), int(sample_shape[2] / k))  # ToDo: GENERALIZAR


    plt.imshow(sample_projected.permute(1, 2, 0).numpy())
    plt.show()


    # train_loader, val_loader = load_dataset('CIFAR10', batch_size=32, train=True, val=True, num_workers=1,
    #                                         pin_memory=True)
