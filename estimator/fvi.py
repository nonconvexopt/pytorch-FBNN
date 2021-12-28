from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import gpytorch

from .inducing import inducing_sampler


__fvi_mode__ = [
    'sampling_based',
    'adversarial',
    'test_point',   
]


class fvi(nn.Module):
    def __init__(
        self,
        variational_model: nn.Module,
        prior: gpytorch.models.GP,
        inducing_strategy: Union[str, inducing_sampler] = 'random',
        fn_noise_std = 1e-8,
    ):
        super(fvi, self).__init__()

        assert hasattr(variational_model, 'weight_mean')
        assert hasattr(variational_model, 'weight_logvar')

        # Variational model is assumed to have pytorch 'forward' method as a __call__ method and 'kld' method
        # returning KL divergence. Also, forward method is assumed to return stochastic results.
        self.variational_model = variational_model
        self.prior = prior
        self.fn_noise_std = fn_noise_std

        if isinstance(mode, str):
            assert inducing_strategy in __fvi_mode__, "Wrong inducing sample strategy specified."
        else:
            self.inducing_strategy = inducing_strategy

        # Check if Variational Module performs stochastic forwarding.
        with torch.no_grad():
            assert self.variational_model(self.inducing_sample()) != self.variational_model(self.inducing_sample())
        
    def inducing_sample(self):
        self.inducing_strategy.sample()

    def fkld(self, x, fn_noise_std = 0.0):
        model_pred_dist = self.model(x)
        gp_pred_dist = self.prior(x)
        
        if fn_noise_std:
            pred_mean = 

        return 
        
    def forward(self, x, inducing_x, repeat = 1):
        try:
            x_num = x.shape[0]
            x = torch.cat([x, inducing_x])
            x = x.unsqueeze(0).repeat(1, repeat)
            inducing_x = inducing_x.unsqueeze(0).repeat(1, repeat)
            self.variational_model(x)

        preds = [self.model(x) for ind in range(repeat)]
        fkld = self.fkld()

        return preds, fkld