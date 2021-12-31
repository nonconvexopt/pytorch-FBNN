from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import gpytorch

from torch_ssge import SSGE


class fvi(nn.Module):
    def __init__(
        self,
        variational_model: nn.Module,
        prior: gpytorch.models.GP,
        grad_estimator: SSGE,
        input_shape = torch.Size(),
        fn_noise_std = 1e-8,
    ):
        super(fvi, self).__init__()

        # Variational model is assumed to have pytorch 'forward' method as a __call__ method and 'kld' method
        # returning KL divergence value with singleton tensor. Also, forward method is assumed to return stochastic results.
        self.variational_model = variational_model
        self.prior = prior
        self.input_shape = input_shape
        self.fn_noise_std = fn_noise_std

        # Check if Variational Module performs stochastic forwarding.
        with torch.no_grad():
            assert torch.ne(self.variational_model(self.inducing_sample()), self.variational_model(self.inducing_sample())).any(), \
                "Variational model should output stochastic results."

        # Spectral Stein Gradient Estimator is used for estimating the gradient of implicit distribution
        # by kernel-based techniques and Nystrom method.
        # Here, I used my implementation of Spectral Stein Gradient Estimator.
        # Please modify this code to apply the other implementation of Spectral Stein Gradient Estimator.
        self.grad_estimator = grad_estimator


    def fkld(self, fs):
        gp_pred_dist = self.prior(xs)
        self.grad_estimator.fit()
        self.grad_estimator()

        if fn_noise_std:
            pred_mean = 

        return 


    def forward(self, x, inducing_x, repeat = 1):
        self.input_shape

        preds = [self.model(x) for ind in range(repeat)]
        fkld = self.fkld()

        return preds, fkld