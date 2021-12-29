from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import gpytorch


class fvi(nn.Module):
    def __init__(
        self,
        variational_model: nn.Module,
        prior: gpytorch.models.GP,
        input_shape = torch.Size(),
        fn_noise_std = 1e-8,
    ):
        super(fvi, self).__init__()

        # Variational model is assumed to have pytorch 'forward' method as a __call__ method and 'kld' method
        # returning KL divergence. Also, forward method is assumed to return stochastic results.
        self.variational_model = variational_model
        self.prior = prior
        self.input_shape = input_shape
        self.fn_noise_std = fn_noise_std

        # Check if Variational Module performs stochastic forwarding.
        with torch.no_grad():
            assert torch.ne(self.variational_model(self.inducing_sample()), self.variational_model(self.inducing_sample())).any(), \
                "Variational model should output stochastic results."


    def fkld(self, xs, fn_noise_std = 0.0):
        model_pred_dist = self.model(xs)
        gp_pred_dist = self.prior(xs)

        if fn_noise_std:
            pred_mean = 

        return 


    def forward(self, x, inducing_x, repeat = 1):
        self.input_shape

        preds = [self.model(x) for ind in range(repeat)]
        fkld = self.fkld()

        return preds, fkld