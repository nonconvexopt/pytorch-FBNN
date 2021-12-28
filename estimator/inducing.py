import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist

class inducing_sampler:
    def sampling(self) -> torch.Tensor:
        raise NotImplementedError


class sampling_based_inducing(inducing_sampler):
    def __init__(self, scalar_dist: dist.distribution.Distribution, input_shape: torch.Size, mean=None, std=None):
        self.scalar_dist = scalar_dist
        self.input_shape = torch.Size(input_shape)

    def sampling(self, sample_size=torch.Size()) -> torch.Tensor:
        shape = torch.Size(sample_size) + self.input_shape
        return self.scalar_dist(shape) * std.expand(input_shape) + mean.expand(input_shape)


class adversarial_inducing(inducing_sampler):
    def __init__(self, dataloader: torch.utils.data.DataLoader):
        self.dataloader = dataloader

    def sampling(self) -> torch.Tensor:
        return 


class test_point_inducing(inducing_sampler):
    def __init__(self, test_loader: torch.utils.data.DataLoader):
        self.test_loader = test_loader

    def sampling(self) -> torch.Tensor:
        return self.test_loader.next()
