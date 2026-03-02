import torch


def coral(source, target):
    """
    Compute CORAL loss (correlation alignment) between source and target features
    """
    d = source.size(1)  # feature dimension
    
    # Source covariance
    source = source - torch.mean(source, 0, keepdim=True)
    source_cov = torch.matmul(source.t(), source) / (source.size(0) - 1)
    
    # Target covariance
    target = target - torch.mean(target, 0, keepdim=True)
    target_cov = torch.matmul(target.t(), target) / (target.size(0) - 1)
    
    # Frobenius norm between source and target covariance
    loss = torch.sum((source_cov - target_cov) ** 2)
    loss = loss / (4 * d * d)
    
    return loss
