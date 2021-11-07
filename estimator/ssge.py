import math
import torch
import gpytorch

class SSGE:
    def __init__(self, kernel: gpytorch.kernels.Kernel, eig_prop_threshold = 0.99, noise = 1e-8):
        self.kernel = kernel
        self.noise = noise
        self.eig_prop_threshold = eig_prop_threshold
        
        self.sample = None
        self.gram = None
        
        self.m = 0
        self.K = None
        self.eigval = None
        self.eigvec = None
        
    def fit(self, x: torch.Tensor) -> None:
        assert x.requires_grad, "'requires_grad' of input tensor must be set to True."
        
        if self.sample is not None:
            self.sample = torch.cat([self.sample, x], axis = 0)
        else:
            self.sample = x

        self.m = self.sample.shape[0]
        self.dim = self.sample.shape[1]
        self.K = self.kernel(self.sample).evaluate()
        if self.noise:
            self.K = self.K + self.noise * torch.eye(self.m, device = self.sample.device)
        
        self.eigval, self.eigvec = torch.linalg.eigh(self.K)
        #Tried to use torch.lobpcg as alternative to torch.linalg.eigh
        #but it seem is not stable yet.
        #self.eigval, self.eigvec = torch.lobpcg(self.K, min(self.m // 3, self.dim))
        with torch.no_grad():
            eig_props = self.eigval.cumsum(-1) / self.eigval.sum(-1, keepdims = True)
            eig_props *= eig_props < self.eig_prop_threshold
            self.j = torch.argmax(eig_props, -1)
        self.eigval = self.eigval[:self.j]
        self.eigvec = self.eigvec[:, :self.j]
        assert (self.eigval > 0).all(), "Kernel matrix is not postive definite."
        
        input_tensor = self.sample.unsqueeze(-1).repeat(1, 1, self.j)
        eigfun_hat = math.sqrt(self.m) * torch.einsum(
            "jnm,mk,k->j",
            self.kernel(torch.einsum("ndj->jnd", input_tensor), self.sample).evaluate(),
            self.eigvec,
            self.eigval.reciprocal()
        )
        #beta should have size d x j
        self.beta = - torch.autograd.grad(
            outputs = eigfun_hat,
            grad_outputs = torch.ones(eigfun_hat.shape, device = eigfun_hat.device),
            inputs = input_tensor,
            retain_graph = True,
        )[0].mean(0)
        
    def grad(self, x: torch.Tensor) -> torch.Tensor:
        assert x.requires_grad, "'requires_grad' of input tensor must be set to True."
        
        K_wing = self.kernel(x, self.sample).evaluate()
        eigfun_hat = math.sqrt(self.m) * torch.einsum("nm,mj->nj", K_wing, self.eigvec) / self.eigval
        gradfun_hat = torch.einsum("nj,mj->nm", eigfun_hat, self.beta)
        return torch.autograd.grad(
            outputs = gradfun_hat,
            grad_outputs = torch.ones(gradfun_hat.shape, device = gradfun_hat.device),
            inputs = x,
        )[0]