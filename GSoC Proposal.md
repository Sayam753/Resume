<h1 style="text-decoration:underline">GSoC Project Proposal</h1>
<h1 style="text-decoration:underline">Add Variational Inference Interface to PyMC4</h1>

## Description

`Variational Inference` is a powerful algorithm that turns the task of computing the posterior(p(z|x)) into an optimization problem.  This project is about implementing two inference algorithms Mean Field ADVI and Full Rank ADVI based on [ADVI](https://arxiv.org/abs/1603.00788) paper in PyMC4. Mean Field ADVI posits a spherical Gaussian family and Full Rank ADVI posits a Multivariate Gaussian family to minimize KL divergence. The implementation will use tf and tfp libraries.

## Interface Design

```python
# Base class for Approximation
class Approximation:
    
    # Defining parameters
    def __init__(self, model, size, random_seed=None):
        self.model = model
        self.size = size
        self.random_seed = random_seed
        # Handle initialization of mu and std
        
    @property
    def mean(self):
        return self.mu

    @property
    def std(self):
        return self.std

    # For Naive Monte Carlo
    def random(self):
        g = tf.random.Generator.from_seed(self.random_seed)
        n = g.normal(shape=some_shape)
        return self.std*n + self.mean
```

```python
class MeanField(Approximation):
    
    def __init__(self, model, size, random_seed):
        super().__init__(model, size, random_seed)
        
    def cov(self):
        sq = tf.math.square(self.std)
        return tf.linalg.diag_part(sq)


class FullRank(Approximation):
    
    def __init__(self, model, size, random_seed):
        super().__init__(model, size, random_seed)
    
    def L(self):
        n = self.size
        entries = n*(n+1)//2
        L = np.zeros([n, n], dtype=int)
        L[np.tril_indices(n)] = np.arange(entries)
        L[np.tril_indices(n)[::-1]] = np.arange(entries)
        return L
        
    def cov(self):
        L = self.L
        return tf.linalg.matmul(L, tf.transpose(L))
```

```python
def fit(model, n=10000, random_seed=None, method='MeanFieldADVI'):
    
    # Transform the model into an unconstrained space
    _, state = pm.evaluate_model_transformed(model)
    logpt = state.collect_log_prob()
        
    # Collect the free random variables
    untransformed = state.untransformed_values
    free_RVs = untransformed.update(state.transformed_values)
    
    # Not sure about the use of local random variables
    size = 0
    for name, dist in free_RVs.items():
        size += int(np.prod(dist.event_shape))
    
    approx = None
    if method == "MeanFieldADVI":
        approx = MeanField(model, size, random_seed)
    else:
        approx = FullRank(model, size, random_seed)

    # Create variational gradient tensor
    q = approx.random()
    elbo = q + tf.reduce_sum(approx.std) + 0.5*size*(1 + tf.math.log(2.0*np.pi))
    
    # Set up optimizer
    
    # Draw samples from variational posterior

    # TODO: Plot the trace using ArviZ
```
