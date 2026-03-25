import numpy as np
from scipy.stats import gamma
from scipy.stats import norm

# import matplotlib.pyplot as plt
# plt.rc('text', usetex=True)
# plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
# plt.rcParams.update({'font.size': 14})

def get_var1_linear(x):
    return 0.5*x + 0.5

def get_var2_linear(x):
    return 2.5 - 2*x

def generate_y_gamma_linear(x):
    return gamma.rvs(get_var1_linear(x), scale=1/get_var2_linear(x))

def generate_y_norm_linear(x):
    return norm.rvs(scale=get_var1_linear(x))

def get_toy_dataset(num_samples=1):
    """
    Generate the target dataset Y ~ N(toy_func(x), get_sd(x)^2)
    """
    x = np.random.uniform(size=num_samples, high=1.0)
    x = np.sort(x)
    x1 = x[np.where(x < 0.33)]
    x2 = x[np.where(np.logical_and(x <= 0.66,x >= 0.33))]
    x3 = x[np.where(x > 0.66)]
    y1 = generate_y_gamma_linear(x1)
    y2 = generate_y_norm_linear(x2)
    y3 = -1*generate_y_gamma_linear(x3)
    y = np.concatenate((y1, y2, y3))
    # y = np.concatenate(y, y3)

    return x, y