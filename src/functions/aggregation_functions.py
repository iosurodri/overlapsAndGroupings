import torch

#####################
# OVERLAP FUNCTIONS #
#####################

def ob_overlap(tensor, keepdim=False, dim=-1):
    return torch.sqrt(torch.min(tensor, keepdim=keepdim, dim=dim)[0] * torch.prod(tensor, keepdim=keepdim, dim=dim) + 1e-10)


def geometric_mean(tensor, keepdim=False, dim=-1):
    return torch.pow(torch.prod(tensor, keepdim=keepdim, dim=dim) + 1e-10, 1/tensor.shape[dim])


######################
# GROUPING FUNCTIONS #
######################

def product_grouping(tensor, keepdim=False, dim=-1):
    return 1 - torch.prod(1 - tensor, keepdim=keepdim, dim=dim)


def minimum_grouping(tensor, keepdim=False, dim=-1):
    return 1 - torch.min(1 - tensor, keepdim=keepdim, dim=dim)[0]
    

def ob_grouping(tensor, keepdim=False, dim=-1):
    return 1 - torch.sqrt(torch.min(1 - tensor, keepdim=keepdim, dim=dim)[0] * torch.prod(1 - tensor, keepdim=keepdim, dim=dim) + 1e-10)


def ob_max_grouping(tensor, keepdim=False, dim=-1):
    return 1 - torch.sqrt(torch.max(tensor, keepdim=keepdim, dim=dim)[0] * torch.prod(1 - tensor, keepdim=keepdim, dim=dim) + 1e-10)


def geometric_grouping(tensor, keepdim=False, dim=-1):
    return 1 - torch.pow(torch.prod(1 - tensor, keepdim=keepdim, dim=dim) + 1e-10, 1/tensor.shape[dim])


def u_grouping(tensor, keepdim=False, dim=-1):
    max_values = torch.max(tensor, keepdim=keepdim, dim=dim)[0]
    return max_values / (max_values + torch.sqrt(1-tensor + 1e-10))
    

available_functions = {
    
}


if __name__ == '__main__':
    print(list_available_functions())
    min_test = choose_aggregation('min')
    sugeno_test = choose_aggregation('sugeno')
    print('Hola')
