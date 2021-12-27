import torch
import torch.nn as nn
import torch.nn.functional as F
import gpytorch

class testpoint_fvi(nn.Module):   
    def __init__(
        self,
        model: nn.Module,
        prior: gpytorch.models.GP,
        fn_noise_std = 1e-8,
        
    ):
        super(testpoint_fvi, self).__init__()
        self.model = model
        self.prior = prior
        self.fn_noise_std = fn_noise_std
        
    def forward(self, x, inducing_x, repeat = 1):
        preds = [self.model(x) for ind in range(repeat)]
        fkld = self.fkld(
            torch.cat([x, inducing_x]),
            self.fn_noise_std,
        )
        
        return preds, fkld