import torch
import torch.nn as nn
import torch.nn.functional as F
import gpytorch

class fvi(nn.Module):
    def __init__(
        self,
        variational_model: nn.Module,
        prior: gpytorch.models.GP,
        fn_noise_std = 1e-8,
    ):
        super(fvi, self).__init__()
        
        assert hasattr(variational_model, 'weigth_mean') and hasattr(variational_model, 'weigth_logvar')
        
        #Variational model is assumed to have pytorch 'forward' method
        #as a __call__ method and 'kld' method returning KLD of parameters.
        #Also, forward method is assumed to return stochastic results.
        self.variational_model = variational_model
        self.prior = prior
        self.fn_noise_std = fn_noise_std
        
    def self.fkld(self, x, fn_noise_std = 0.0):
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