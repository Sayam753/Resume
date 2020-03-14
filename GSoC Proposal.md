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

## Contributing to PyMC4
1. Pull Request [#220](https://github.com/pymc-devs/pymc4/pull/220) (Merged): Add AutoRegressive distribution - <br/>
This PR added Auto Regressive distribution by wrapping `sts.AutoRegressive` Model. The main task was to call `make_state_space_model` method with suitable arguments to capture the underlying the `tfd.LinearGaussianStateSpaceModel`. It took a lot of debugging to make this AR class compatible with PyMC4.

2. Pull Request [#215](https://github.com/pymc-devs/pymc4/pull/215) (Merged): Add default transform(sigmoid) for Unit Continuous Distribution - <br/>
This PR added sigmoid transform to Unit Continuous Distribution. To make the default transform compatible with PyMC4, I also added Sigmoid transform that used `tfb.Sigmoid` bijector.

3. Pull Request [#212](https://github.com/pymc-devs/pymc4/pull/212) (Merged): Update design_guide notebook - <br/>
This small PR fixed typos and variable names in `pymc4_design_guide.ipynb`.

4. Issue [#211](https://github.com/pymc-devs/pymc4/issues/211) (Closed): Installation issues
I encountered installation issues while setting up the working environment using pip. So, I created the issue and Luciano Paz helped me out with other ways of installing PyMC4.

## Personal Projects
1. Send to S3 - [Github](https://github.com/Sayam753/SendToS3) <br/>
This python project sends backup files to AWS S3 bucket using Boto3. Searching for files is done by regex and results of logs are sent to email using smtplib.

2. Osint-Spy - [Github](https://github.com/Sayam753/OSINT-SPY) <br/>
This Python project performs Osint scan on email, domain, ip, organization, etc.
This information can be used by Data Miners or Penetration Testers in order to find deep information about their target.

3. Turbofan Degradation - [Colab](https://colab.research.google.com/drive/1sCZcJSmRarYbQKDYeaqiLnzXyzFolRC0) <br/>
Implemented a Deep learning based Encoder-Decoder model ([paper](https://www.researchgate.net/publication/336150924_A_Novel_Deep_Learning-Based_Encoder-Decoder_Model_for_Remaining_Useful_Life_Prediction)) for analysing the turbofan degradation dataset provided by NASA.

4. Neural Network from Scratch - [Colab](https://colab.research.google.com/drive/1iU38tTeEvUI_sjt6vVAuhedMWOPUdr5E) <br/>
Implemented a deep neural network from scratch in numpy with custom hyperparameters.

## Basic/Contact Information

- Time Zone: UTC+05:30
- Github Handle: [Sayam753](https://github.com/Sayam753)
- Resume: [Google drive link](https://drive.google.com/file/d/1mrNC3qtieWKH1i2mhqH6xiFCt-EwGJ0b/view?usp=sharing), [Github link](https://github.com/Sayam753/Resume)
- Contact details: [Gmail](sayamkumar049@gmail.com), [Yahoo](sayamkumar753@yahoo.in), [LinkedIn](https://www.linkedin.com/in/sayam049/), [Twitter](https://twitter.com/sayamkumar753), +91 9815247310 (Mobile)

### Personal Info

I am Sayam Kumar from Indian Institute of Information Technology Sri City, India. I am a second year Undergraduate pursuing a Bachelor's in Computer Science Engineering. I mostly code in Python. I am interested to work on the project to expand my knowledge in Machine Learning and Bayesian Statistics. With my continuous efforts to learn and know more about Bayesian Statistics, I believe I will be able to complete the project in time. Also this is my first time participating in GSoC.

### Commitments

As I have no other projects/internships planned for summers, I can spend 40~50 hours per week or more if required working on the project. Along the way, I will design progress reports and extensive documentation of the implementation of various classes. This will help in submitting reports to mentor and Google at evaluation time.

## References and Useful Papers

- [Automatic Differentiation Variational Inference](https://arxiv.org/pdf/1603.00788.pdf). Kucukelbir, A., Tran, D., Ranganath, R., Gelman, A., and Blei, D. M. (2016).
- [Automatic Variational Inference in Stan](https://arxiv.org/abs/1506.03431). Kucukelbir, A., Ranganath, R., Gelman, A., & Blei, D. (2015).
- [Operator Variational Inference](https://arxiv.org/abs/1610.09033). Rajesh Ranganath, Jaan Altosaar, Dustin Tran, David M. Blei (2016).
- [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114). Kingma, D. P., & Welling, M. (2014).
