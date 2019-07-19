import torch


def mse_metric(output, target):
    with torch.no_grad():
        assert pred.shape[0] == len(target)
        accuracy = torch.mean((output - target)**2, dim=0)
    return accuracy
