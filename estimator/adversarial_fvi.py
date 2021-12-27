import torch
import torch.nn as nn
import torch.nn.functional as F
import gpytorch

class fvi(nn.Module):   
    def __init__(
        self,
        model: nn.Module,
        estimator, 
        prior: gpytorch.models.GP,
        
    ):
        super(fvi, self).__init__()
        self.model = model
        self.estimator = estimator
        self.prior = prior

        self.noise = nn.Parameter(inducing_y.std(-1), device = device)
        
        self.kld_scale = kld_scale
        
    def forward(self, x, inducing_x, repeat = 1):
        preds = [self.model(x) for ind in range(repeat)]
        
        prior_samples = self.prior(x)
        preds = 
        
        return 