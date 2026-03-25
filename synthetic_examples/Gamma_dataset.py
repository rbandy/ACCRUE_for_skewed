import numpy as np
from scipy.stats import gamma
from scipy.stats import norm

# import matplotlib.pyplot as plt
# plt.rc('text', usetex=True)
# plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
# plt.rcParams.update({'font.size': 14})

def get_var1_trig(x):
    return np.exp(np.sin(2*np.pi*x))/3

def get_var2_trig(x):
    return np.cos(2*np.pi*x) + 2
    # return (np.cos(2*np.pi*x) +2)/3
def generate_y_gamma_trig(x):
    return gamma.rvs(get_var1_trig(x), scale=1/get_var2_trig(x))

def get_toy_dataset(num_samples=1, x=None):
    """
    Generate the target dataset Y ~ N(toy_func(x), get_sd(x)^2)
    """
    if x is None:
        x = np.random.uniform(size=num_samples, high=1.0)
    y = -1*generate_y_gamma_trig(x)

    return x, y