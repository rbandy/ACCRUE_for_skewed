import numpy as np
from scipy.stats import laplace_asymmetric
from twopiece.scale import *
from twopiece.shape import *
from twopiece.double import *

# toy problem x \in [0,1], f(x) = 0
# Error dist: f(x) + AL(kappa(x), 1/lambda(x)) or
#             f(x) + TPG(sigma1(x), sigma2(x)) 

def get_sigma1_linear(x):
    """
    compute left-side std for the toy function
    """
    return 0.5*x + 0.5

def get_sigma2_linear(x):
    """
    compute right-side std for the toy function
    """
    return 2.5 - 2*x

def get_kappa_linear(x):
    """
    compute asymmetry for the toy function
    """
    return 0.5*x + 0.5

def get_inverse_lam_linear(x):
    """
    compute scale for the toy function
    """
    return 1/(2.5 - 2*x)

def get_var1_trig(x):
    return np.exp(np.sin(2*np.pi*x))/3

# def get_var2_trig(x):
#     return np.cos(x)
def get_var2_trig(x):
    return np.cos(2*np.pi*x) + 2
    # return (np.cos(2*np.pi*x) +2)/3

def generate_y_TPG_linear(x):
    y = np.zeros(len(x))
    for i in range(len(x)):
        dist = tpnorm(loc=0.0, sigma1=get_sigma1_linear(x[i]), sigma2=get_sigma2_linear(x[i]))
        y[i] = dist.random_sample(size=1)[0]
    return y

def generate_y_AL_linear(x):
    return laplace_asymmetric.rvs(get_kappa_linear(x), scale=get_inverse_lam_linear(x))

def generate_y_TPG_trig(x):
    y = np.zeros(len(x))
    for i in range(len(x)):
        dist = tpnorm(loc=0.0, sigma1=get_var1_trig(x[i]), sigma2=get_var2_trig(x[i]))
        y[i] = dist.random_sample(size=1)[0]
    return y

def generate_y_AL_trig(x):
    return laplace_asymmetric.rvs(get_var1_trig(x), scale=1/get_var2_trig(x))

def generate_y_TPG_linear_trig(x):
    y = np.zeros(len(x))
    for i in range(len(x)):
        dist = tpnorm(loc=0.0, sigma1=get_sigma1_linear(x[i]), sigma2=get_var2_trig(x[i]))
        y[i] = dist.random_sample(size=1)[0]
    return y

def generate_y_AL_linear_trig(x):
    return laplace_asymmetric.rvs(get_kappa_linear(x), scale=1/get_var2_trig(x))

def get_toy_dataset(dist, dataset, num_samples=1):
    """
    Generate the target dataset Y ~ N(toy_func(x), get_sd(x)^2)
    """
    x = np.random.uniform(size=num_samples, high=1.0)
    x = np.sort(x)
    if dataset=="NO":
        if dist == "AL":
            y = generate_y_AL_linear(x)
        elif dist == "TPG":
            y = generate_y_TPG_linear(x)
        else:
            raise NameError("Undefined distribution!")
    elif dataset=="NO_trig":
        if dist == "AL":
            y = generate_y_AL_trig(x)
        elif dist == "TPG":
            y = generate_y_TPG_trig(x)
        else:
            raise NameError("Undefined distribution!")
    elif dataset=="NO_linear_trig":
        if dist == "AL":
            y = generate_y_AL_linear_trig(x)
        elif dist == "TPG":
            y = generate_y_TPG_linear_trig(x)
        else:
            raise NameError("Undefined distribution!")
    else:
            raise NameError("Undefined dataset!")
    return x, y