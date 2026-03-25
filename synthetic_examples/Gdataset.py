import numpy as np
from scipy.stats import laplace_asymmetric
from twopiece.scale import *
from twopiece.shape import *
from twopiece.double import *

# toy problem from: https://ecamporeale.github.io/papers/IJUQ1104(5)-34623.pdf
# G dataset: x \in [0,1], f(x) = 2*sin(2*\pi*x)
# 

def toy_func(x):
    """
    Generate f(x) aka dataset G
    """
    # sanity check: x \in [0,1]
    if np.any(x > 1.0) or np.any(x < 0.0):
        raise AssertionError("Input out of range!", i)
    return 2.0*np.sin(2.0*np.pi*x)

def get_sigma1(x):
    """
    compute left-side std for the toy function
    """
    return 0.5*x + 0.5

def get_sigma2(x):
    """
    compute right-side std for the toy function
    """
    return 2.5 - 2*x

def get_kappa(x):
    """
    compute asymmetry for the toy function
    """
    return 0.5*x + 0.5

def get_inverse_lam(x):
    """
    compute scale for the toy function
    """
    return 1/(2.5 - 2*x)

def generate_y_TPG(x):
    y = np.zeros(len(x))
    for i in range(len(x)):
        dist = tpnorm(loc=0.0, sigma1=get_sigma1(x[i]), sigma2=get_sigma2(x[i]))
        y[i] = dist.random_sample(size=1)[0]
    return toy_func(x) + y

def generate_y_AL(x):
    return toy_func(x) + laplace_asymmetric.rvs(get_kappa(x), scale=get_inverse_lam(x))

def get_toy_dataset(dist, num_samples=1):
    """
    Generate the target dataset Y ~ N(toy_func(x), get_sd(x)^2)
    """
    x = np.random.uniform(size=num_samples, high=1.0)
    x = np.sort(x)
    if dist == "AL":
        y = generate_y_AL(x)
    elif dist == "TPG":
        y = generate_y_TPG(x)
    else:
        raise NameError("Undefined distribution!")
    return x, y