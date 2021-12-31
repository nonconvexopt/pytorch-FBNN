import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from torch.utils.data import Dataset


class inducing_sampler(nn.Module):
    # Inherited nn.Module to gain performance benefit by torch.jit.script
    def forward(self, sample_size=torch.Size(), batch: torch.Tensor) -> torch.Tensor:
        # Method Sampling inducing points.
        # `batch` argument may not be used.
        raise NotImplementedError


class point_inducing(inducing_sampler):
    def __init__(self):
        super(point_inducing, self).__init__()

    def forward(self, num_inducing=torch.Size(), batch: torch.Tensor) -> torch.Tensor:
        num_inducing = torch.Size(num_inducing)
        return batch.expand(num_inducing + batch.shape)


class sampling_based_inducing(inducing_sampler):
    def __init__(self, sample_dist: dist.distribution.Distribution, input_shape: torch.Size, mean=None, std=None):
        self.sample_dist = sample_dist
        self.input_shape = torch.Size(input_shape)
        assert (mean is None) == (std is None), "Please specify both mean and std if one of them should be given."
        self.mean, self.std = mean, std

    def forward(self, num_inducing=torch.Size(), batch: torch.Tensor) -> torch.Tensor:
        shape = torch.Size(num_inducing) + self.input_shape
        sample = self.sample_dist(shape)
        if self.mean is not None:
            sample = sample * std.expand(shape) + mean.expand(shape)
        return sample


class adversarial_inducing(inducing_sampler):
    def __init__(self, sample_dist: dist.distribution.Distribution, latent_shape: torch.Size, generator: nn.Module):
        self.sample_dist = sample_dist
        self.latent_shape = torch.Size(latent_shape)
        self.generator = generator

    def forward(self, num_inducing=torch.Size(), batch: torch.Tensor) -> torch.Tensor:
        num_inducing = torch.Size(num_inducing)
        z = sample_dist.sample(num_inducing + self.latent_shape)
        return self.generator(z)
