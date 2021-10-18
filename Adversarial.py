import torch
import torch.nn as nn
import torch.nn.functional as F
import gpytorch

class fvi_prior(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(fvi_prior, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
    
class fvi(nn.Module):
    class GP_prior(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood, kld_scale):
            super(GP_prior, self).__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ConstantMean()
            self.cov_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(
                    ard_num_dims = train_x.shape[-1]
                )
            )
            self.cov_module.initialize_from_data(train_x, train_y)
            
        def forward(self, x):
            return gpytorch.distributions.MultivariateNormal(
                self.mean_module(x),
                self.cov_module(x),
            )
    
    def __init__(self, model: nn.Module, inducing_x, inducing_y, likelihood):
        super(fvi, self).__init__()
        self.model = model
        
        self.prior = GP_prior(
            inducing_x,
            inducing_y,
            likelihood
        )
        
        self.posterior = 
        
        self.noise = nn.Parameter(inducing_y.std(-1), device = device) 
        
        self.kld_scale = kld_scale
        
        
    def forward(self, x, inducing_x):
        prior_samples = self.prior(x)
        posterior_samples = self.posterior(x, inducing_x)
        
        return 