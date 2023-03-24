import torch


class LayerNorm:
    """ Ensures that each layer across all batches (rows) has a unit gaussian distribution - mean 0, std 1"""
    
    def __init__(self, dim, eps=1e-5):
        self.eps = eps
        # parameters trained with backprop
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)
        
        
    def __call__(self, x):
        xmean = x.mean(1, keepdim=True) # batch mean
        xvar = x.var(1,  keepdim=True) # batch variance
     
        # normalize to unit variance
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps)
        self.out = self.gamma * xhat + self.beta
        
        return self.out
    
    def parameters(self):
        return [self.gamma, self.beta]
    
torch.manual_seed(1337)
module = LayerNorm(100)
x = torch.randn(32, 100) # batch of 32, 100-dimensional vectors
x = module(x)


# mean, std of one feature across all inputs
print("Sample usage of BatchNorm1d,  mean across rows should be close to zero, standard deviation close to 1")
print(f"Normalized mean: {x[0, :].mean():.2f}, Normalized std: {x[0, :].std():.2f}")
print(f"Columns are not normalized; mean: {x[:, 0].mean():.2f}, std: {x[:,0].std():.2f}")

            