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


# def ob_max_grouping(tensor, keepdim=False, dim=-1):
#     return 1 - torch.sqrt(torch.max(tensor, keepdim=keepdim, dim=dim)[0] * torch.prod(1 - tensor, keepdim=keepdim, dim=dim) + 1e-10)


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
    out_tensor = torch.pow(tensor+1e-10, p)  # Less efficient than computing the power of the single greatest value, but more generalizable interface
    return torch.max(out_tensor, keepdim=keepdim, dim=dim)[0]

def product_power_grouping(tensor, p, keepdim=False, dim=-1):
    out_tensor = torch.pow((1 - tensor)+1e-10, p)
    out_tensor = 1 - torch.prod(out_tensor, keepdim=keepdim, dim=dim)
    return out_tensor

def geometric_power_grouping(tensor, p, keepdim=False, dim=-1):
    out_tensor = torch.pow((1 - tensor)+1e-10, p)
    out_tensor = 1 - torch.pow(torch.prod(out_tensor, keepdim=keepdim, dim=dim) + 1e-10, 1/tensor.shape[dim])  # prod(X) ** (1/n)
    return out_tensor

####################
# T-NORM FUNCTIONS #
####################

def lukasiewicz_tnorm(tensor, keepdim=False, dim=-1):
    tensor_shape = list(tensor.shape)
    tensor_shape.pop(dim)
    out_tensor = tensor.new_zeros(tensor_shape)
    zero = tensor.new_zeros([1])
    if (dim == -1) or (dim == len(tensor.shape)-1):
        out_tensor = tensor[..., 0]
        for i in range(1, tensor.shape[dim]):
            out_tensor = torch.maximum(out_tensor + tensor[..., i] - 1, zero)
    else:
        # More general implementation using index_select (when dimension is unknown)
        out_tensor = torch.index_select(tensor, dim, tensor.new_tensor([0], dtype=torch.int)).squeeze(dim)
        for i in range(1, tensor.shape[dim]):
            out_tensor = torch.maximum(out_tensor + torch.index_select(tensor, dim, tensor.new_tensor([i], dtype=torch.int)).squeeze(dim) - 1, zero)
    if keepdim:
        torch.unsqueeze(out_tensor, dim=dim)
    return out_tensor


def hamacher_tnorm(tensor, keepdim=False, dim=-1):
    tensor_shape = list(tensor.shape)
    tensor_shape.pop(dim)
    out_tensor = tensor.new_zeros(tensor_shape)
    zeros = tensor.new_zeros(tensor_shape)
    if (dim == -1) or (dim == len(tensor.shape)-1):
        out_tensor = tensor[..., 0]
        for i in range(1, tensor.shape[dim]):
            diff_indices = torch.where(torch.abs(out_tensor - tensor[..., i]) > 1e-9)
            prev_tensor = out_tensor
            out_tensor = zeros
            out_tensor[diff_indices] = torch.mul(prev_tensor[diff_indices], tensor[..., i][diff_indices]) / (
                prev_tensor[diff_indices] + tensor[..., i][diff_indices] - torch.mul(prev_tensor[diff_indices], tensor[..., i][diff_indices]))
    else:
        # More general implementation using index_select (when dimension is unknown)
        out_tensor = torch.index_select(tensor, dim, tensor.new_tensor([0], dtype=torch.int)).squeeze(dim)
        for i in range(1, tensor.shape[dim]):
            out_tensor = torch.maximum(out_tensor + torch.index_select(tensor, dim, tensor.new_tensor([i], dtype=torch.int)).squeeze(dim) - 1, zero)
    if keepdim:
        torch.unsqueeze(out_tensor, dim=dim)
    return out_tensor
