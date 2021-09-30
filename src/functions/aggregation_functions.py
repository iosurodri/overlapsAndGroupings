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
    out_tensor = torch.max(tensor, keepdim=keepdim, dim=dim)[0]
    return out_tensor / (out_tensor + torch.pow(torch.prod(1-tensor, keepdim=keepdim, dim=dim) + 1e-10, 1/tensor.shape[dim]))
    
#################################
# GROUPING FUNCTIONS WITH POWER #
#################################

# Note: All values to be powered are added a value epsilon in order to avoid instabilities in the backward pass

def max_power_grouping(tensor, p, keepdim=False, dim=-1):
    out_tensor = torch.max(tensor, keepdim=keepdim, dim=dim)[0]
    return torch.pow(out_tensor+1e-10, p)

def product_power_grouping(tensor, p, keepdim=False, dim=-1):
    out_tensor = torch.pow((1 - tensor)+1e-10, p)
    out_tensor = 1 - torch.prod(out_tensor, keepdim=keepdim, dim=dim)
    return out_tensor

def geometric_power_grouping(tensor, p, keepdim=False, dim=-1):
    out_tensor = torch.pow((1 - tensor)+1e-10, p)
    out_tensor = 1 - torch.pow(torch.prod(out_tensor, keepdim=keepdim, dim=dim) + 1e-10, 1/tensor.shape[dim])  # prod(X) ** (1/n)
    return out_tensor

available_functions = {
    
}


if __name__ == '__main__':
    print(list_available_functions())
    min_test = choose_aggregation('min')
    sugeno_test = choose_aggregation('sugeno')
    print('Hola')
