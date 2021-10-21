import torch
import torch.nn as nn
import torch.nn.functional as F
import gpytorch

class fvi(nn.Module):   
    def __init__(
        self,
        model: nn.Module,
        prior: gpytorch.models.GP,
        fn_noise_std = 1e-8,
    ):
        super(fvi, self).__init__()
        self.model = model
        self.prior = prior
        self.fn_noise_std = fn_noise_std
        
    def self.fkld(self, x, fn_noise_std = 0.0):
        model_pred_dist = self.model(x)
        gp_pred_dist = self.prior(x)
        
        if fn_noise_std:
            pred_mean = 
        
        return 
        
    def forward(self, x, inducing_x, repeat = 1):
        preds = [self.model(x) for ind in range(repeat)]
        fkld = self.fkld(torch.cat([x, inducing_x]))
        
        return preds, fkld