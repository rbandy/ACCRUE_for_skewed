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
def generate_y_tukey_trig(x):
    N=len(x)
    tuk = np.zeros(N)
    z = norm.rvs(size=N)
    g = get_var1_trig(x)
    h = get_var2_trig(x)
    for i in range(N):
        tuk[i] = ((np.exp(g[i]*z[i]) - 1)/g[i])*np.exp(h[i]*z[i]**2/2)
    return tuk

def get_toy_dataset(num_samples=1):
    """
    Generate the target dataset Y ~ N(toy_func(x), get_sd(x)^2)
    """
    x = np.random.uniform(size=num_samples, high=1.0)
    y = generate_y_tukey_trig(x)

    return x, y