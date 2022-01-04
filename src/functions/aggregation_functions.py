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

# Hamacher product:
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
            # if a == b -> T(a, b) = 0
            out_tensor = zeros  
            # otherwise -> T(a, b) = ab / (a + b + ab)
            out_tensor[diff_indices] = torch.mul(prev_tensor[diff_indices], tensor[..., i][diff_indices]) / (
                prev_tensor[diff_indices] + tensor[..., i][diff_indices] - torch.mul(prev_tensor[diff_indices], tensor[..., i][diff_indices]))
    else:
        # TODO: Replace this code by the implementation of hamacher t-norm
        # More general implementation using index_select (when dimension is unknown)
        out_tensor = torch.index_select(tensor, dim, tensor.new_tensor([0], dtype=torch.int)).squeeze(dim)
        for i in range(1, tensor.shape[dim]):
            out_tensor = torch.maximum(out_tensor + torch.index_select(tensor, dim, tensor.new_tensor([i], dtype=torch.int)).squeeze(dim) - 1, zero)
    if keepdim:
        torch.unsqueeze(out_tensor, dim=dim)
    return out_tensor

######################
# T-CONORM FUNCTIONS #
######################

def probabilistic_sum(tensor, keepdim=False, dim=-1):
    tensor_shape = list(tensor.shape)
    tensor_shape.pop(dim)
    out_tensor = tensor.new_zeros(tensor_shape)
    if (dim == -1) or (dim == len(tensor.shape)-1):
        out_tensor = tensor[..., 0]
        for i in range(1, tensor.shape[dim]):
            out_tensor = out_tensor + tensor[..., i] - torch.mul(out_tensor, tensor[..., i])
    else:
        # More general implementation using index_select (when dimension is unknown)
        out_tensor = torch.index_select(tensor, dim, tensor.new_tensor([0], dtype=torch.int)).squeeze(dim)
        for i in range(1, tensor.shape[dim]):
            out_tensor = out_tensor + torch.index_select(tensor, dim, tensor.new_tensor([i], dtype=torch.int)).squeeze(dim) - torch.mul(
                out_tensor, torch.index_select(tensor, dim, tensor.new_tensor([i], dtype=torch.int)).squeeze(dim))
    if keepdim:
        torch.unsqueeze(out_tensor, dim=dim)
    return out_tensor

# Dual to the Lukasiewicz t-norm
def bounded_sum(tensor, keepdim=False, dim=-1):
    tensor_shape = list(tensor.shape)
    tensor_shape.pop(dim)
    out_tensor = tensor.new_zeros(tensor_shape)
    one = tensor.new_ones([1])
    if (dim == -1) or (dim == len(tensor.shape)-1):
        out_tensor = tensor[..., 0]
        for i in range(1, tensor.shape[dim]):
            out_tensor = torch.minimum(out_tensor + tensor[..., i], one)
    else:
        # More general implementation using index_select (when dimension is unknown)
        out_tensor = torch.index_select(tensor, dim, tensor.new_tensor([0], dtype=torch.int)).squeeze(dim)
        for i in range(1, tensor.shape[dim]):
            out_tensor = torch.minimum(out_tensor, torch.index_select(tensor, dim, tensor.new_tensor([i], dtype=torch.int)).squeeze(dim), one)
    if keepdim:
        torch.unsqueeze(out_tensor, dim=dim)
    return out_tensor

# Dual to the hamacher product:
def hamacher_tconorm(tensor, keepdim=False, dim=-1):
    tensor_shape = list(tensor.shape)
    tensor_shape.pop(dim)
    out_tensor = tensor.new_zeros(tensor_shape)
    ones = tensor.new_ones(tensor_shape)
    if (dim == -1) or (dim == len(tensor.shape)-1):
        out_tensor = tensor[..., 0]
        for i in range(1, tensor.shape[dim]):
            diff_indices = torch.where(torch.abs(torch.mul(out_tensor, tensor[..., i]) - 1) > 1e-9)
            prev_tensor = out_tensor
            # if ab == 1 -> T(a, b) = 1
            out_tensor = ones  
            # otherwise -> T(a, b) = (2ab - a - b) / (ab - 1)
            out_tensor[diff_indices] = (
                2 * torch.mul(prev_tensor[diff_indices], tensor[..., i][diff_indices]) - prev_tensor[diff_indices] - tensor[..., i][diff_indices]) / (
                torch.mul(prev_tensor[diff_indices], tensor[..., i][diff_indices]) - 1)
    else:
        # TODO: Replace this code by the implementation of hamacher t-norm
        # More general implementation using index_select (when dimension is unknown)
        out_tensor = torch.index_select(tensor, dim, tensor.new_tensor([0], dtype=torch.int)).squeeze(dim)
        for i in range(1, tensor.shape[dim]):
            out_tensor = torch.maximum(out_tensor + torch.index_select(tensor, dim, tensor.new_tensor([i], dtype=torch.int)).squeeze(dim) - 1, zero)
    if keepdim:
        torch.unsqueeze(out_tensor, dim=dim)
    return out_tensor

# Dual to the Hamacher t-norm (p=2)
def einstein_sum(tensor, keepdim=False, dim=-1):
    tensor_shape = list(tensor.shape)
    tensor_shape.pop(dim)
    out_tensor = tensor.new_zeros(tensor_shape)
    if (dim == -1) or (dim == len(tensor.shape)-1):
        out_tensor = tensor[..., 0]
        for i in range(1, tensor.shape[dim]):
            out_tensor = (out_tensor + tensor[..., i]) / (1 + torch.mul(out_tensor, tensor[..., i]))
    else:
        # More general implementation using index_select (when dimension is unknown)
        out_tensor = torch.index_select(tensor, dim, tensor.new_tensor([0], dtype=torch.int)).squeeze(dim)
        for i in range(1, tensor.shape[dim]):
            out_tensor = (out_tensor + torch.index_select(tensor, dim, tensor.new_tensor([i], dtype=torch.int)).squeeze(dim)) / (
                1 + torch.mul(out_tensor, torch.index_select(tensor, dim, tensor.new_tensor([i], dtype=torch.int)).squeeze(dim)))
    if keepdim:
        torch.unsqueeze(out_tensor, dim=dim)
    return out_tensor

#####################
# UNINORM FUNCTIONS #
#####################

def uninorm_min_max(tensor, keepdim=False, dim=-1, threshold=0.5):
    tensor_shape = list(tensor.shape)
    tensor_shape.pop(dim)
    out_tensor = tensor.new_zeros(tensor_shape)
    if (dim == -1) or (dim == len(tensor.shape)-1):
        out_tensor = tensor[..., 0]
        for i in range(1, tensor.shape[dim]):
            tnorm_tensor = torch.minimum(out_tensor, tensor[..., i])
            tconorm_tensor = torch.maximum(out_tensor, tensor[..., i])
            out_tensor = torch.where(torch.logical_or(out_tensor >= threshold, tensor[..., i] >= threshold), tconorm_tensor, tnorm_tensor)
    else:
        # More general implementation using index_select (when dimension is unknown)
        out_tensor = torch.index_select(tensor, dim, tensor.new_tensor([0], dtype=torch.int)).squeeze(dim)
        for i in range(1, tensor.shape[dim]):
            out_tensor = (out_tensor + torch.index_select(tensor, dim, tensor.new_tensor([i], dtype=torch.int)).squeeze(dim)) / (
                1 + torch.mul(out_tensor, torch.index_select(tensor, dim, tensor.new_tensor([i], dtype=torch.int)).squeeze(dim)))
    if keepdim:
        torch.unsqueeze(out_tensor, dim=dim)
    return out_tensor

def uninorm_product(tensor, keepdim=False, dim=-1, threshold=0.5):
    tensor_shape = list(tensor.shape)
    tensor_shape.pop(dim)
    out_tensor = tensor.new_zeros(tensor_shape)
    if (dim == -1) or (dim == len(tensor.shape)-1):
        out_tensor = tensor[..., 0]
        for i in range(1, tensor.shape[dim]):
            tnorm_tensor = 2 * torch.multiply(out_tensor, tensor[..., i])
            tconorm_tensor = 2 * (out_tensor + tensor[..., i] - torch.multiply(out_tensor, tensor[..., i])) - 1
            prev_tensor = out_tensor
            out_tensor = torch.maximum(out_tensor, tensor[..., i])
            out_tensor = torch.where(torch.logical_and(prev_tensor <= threshold, tensor[..., i] <= threshold), tnorm_tensor, out_tensor)
            out_tensor = torch.where(torch.logical_and(prev_tensor >= threshold, tensor[..., i] >= threshold), tconorm_tensor, out_tensor)
    else:
        # More general implementation using index_select (when dimension is unknown)
        out_tensor = torch.index_select(tensor, dim, tensor.new_tensor([0], dtype=torch.int)).squeeze(dim)
        for i in range(1, tensor.shape[dim]):
            out_tensor = (out_tensor + torch.index_select(tensor, dim, tensor.new_tensor([i], dtype=torch.int)).squeeze(dim)) / (
                1 + torch.mul(out_tensor, torch.index_select(tensor, dim, tensor.new_tensor([i], dtype=torch.int)).squeeze(dim)))
    if keepdim:
        torch.unsqueeze(out_tensor, dim=dim)
    return out_tensor