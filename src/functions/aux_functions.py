import torch

#########################
# Normalization methods #
#########################

def min_max_normalization(tensor):
    min_value = tensor.min()
    max_value = tensor.max()
    normalized_tensor = (tensor - min_value) / (max_value - min_value)
    normalization_params = {
        'min': min_value,
        'max': max_value
    }
    return normalized_tensor, normalization_params


def quantile_normalization(tensor):
    quantile_bottom = torch.quantile(tensor, q=0.05)
    quantile_top = torch.quantile(tensor, q=0.95)
    normalized_tensor = tensor
    normalized_tensor[tensor <= quantile_bottom] = 0
    normalized_tensor[tensor >= quantile_top] = 1
    normalized_tensor[(tensor > quantile_bottom) & (tensor < quantile_top)] = ((tensor > quantile_bottom) and (tensor < quantile_top) - quantile_bottom) / (quantile_top - quantile_bottom)
    normalization_params = {
        'min': quantile_bottom,
        'max': quantile_top
    }
    return normalized_tensor, normalization_params


#########################
# Denormalization methods #
#########################

def min_max_denormalization(tensor, normalization_params):
    min_value = normalization_params['min']
    max_value = normalization_params['max']
    denormalized_tensor = tensor * (max_value - min_value) + min_value
    return denormalized_tensor

def sigmoid_denormalization(tensor, _=None):
    return torch.log(tensor / (1 - tensor + 1e-10))

if __name__ == '__main__':
    random_tensor = torch.rand([32, 64, 16, 16], dtype=torch.float)
    reduced_tensor = pca_reduction(random_tensor, [1, 2], reduction_factor=0.5)

    covariance_matrix = cov_tensor(random_tensor, [2, 1])
    test = torch.symeig(covariance_matrix, eigenvectors=True)
    print(covariance_matrix.shape)

