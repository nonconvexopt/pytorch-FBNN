import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist


class inducing_sampler:
    # TODO: inherit PyTorch Dataloader
    input_shape = torch.Size()


class sampling_based_inducing(inducing_sampler):
    def __init__(self, sample_dist: dist.distribution.Distribution, input_shape: torch.Size, mean=None, std=None):
        self.sample_dist = sample_dist
        self.input_shape = torch.Size(input_shape)

    def sampling(self, sample_size=torch.Size(), batch: torch.Tensor) -> torch.Tensor:
        shape = torch.Size(sample_size) + self.input_shape
        return self.sample_dist(shape) * std.expand(shape) + mean.expand(shape)


class adversarial_inducing(inducing_sampler):
    def __init__(self, sample_dist: dist.distribution.Distribution, latent_shape: torch.Size, generator: nn.Module):
        self.latent_shape = torch.Size(latent_shape)
        self.generator = generator

    def sampling(self, sample_size=torch.Size(), batch: torch.Tensor) -> torch.Tensor:
        shape = torch.Size(sample_size) + self.latent_shape
        return self.generator(shape)


class test_point_inducing(inducing_sampler):
    def __init__(self, test_loader: torch.utils.data.DataLoader):
        self.test_loader = test_loader

    def sampling(self, sample_size=torch.Size(), batch: torch.Tensor) -> torch.Tensor:
        return self.test_loader.next()
