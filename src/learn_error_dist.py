import sys
import multiprocessing
from itertools import repeat
import pickle
import os

import random
import numpy as np
import scipy
from scipy import interpolate

from skorch import NeuralNetRegressor
from skorch import NeuralNet
from skorch import callbacks
from skorch import dataset as skorch_dataset
from skorch.callbacks import Checkpoint
import torch
from torch import nn
from sklearn.preprocessing import MinMaxScaler # data preparation

# Asymmetric Laplace imports:
import asymmLaplace_accrue_torch
import NN_regression_AL
from scipy.stats import laplace_asymmetric

# Two-piece Gaussian imports:
import twoPieceGauss_accrue_torch
import NN_regression_TPG
from twopiece.scale import *
from twopiece.shape import *
from twopiece.double import *

sys.path.insert(0, "../synthetic_examples/")
import NOdataset
import Franken_dataset
import Gamma_dataset
import Tukey_dataset

def get_data(dist, dataset="NO", N=1000, x=None):
    """
    define distribution to generate error and learn parameters in ACCRUE
    dist:
        "AL": asymmetric Laplace with parameters kappa and 1/lambda
        "TPG": two-piece Gaussian with parameters sigma1 and sigma 2
    dataset:
        "G": x \in [0,1], f(x)=2sin(2\pi x)
        "NO": x \in [0,1], f(x)=0 w/ linear params
        "NO_trig": x \in [0,1], f(x)=0 w/ trig params
        "NO_linear_trig": x \in [0,1], f(x)=0 w/ linear/trig params
    N: number of samples to generate
    outputs:
        x: list of N input samples (ordered)
        y: list of N noisy observations from f(x) with dist as the noise 
           distribution
        f: list of N f(x) predictions
    """
    y = None
    f = None
    if dataset == "G":
        x, y = Gdataset.get_toy_dataset(dist, num_samples=N)
        f = Gdataset.toy_func(x)
    elif "NO" in dataset:
        x, y = NOdataset.get_toy_dataset(dist, dataset, num_samples=N)
        f = np.zeros(len(x))
    elif "Franken" in dataset:
        x, y = Franken_dataset.get_toy_dataset(num_samples=N)
        f = np.zeros(len(x))
    elif "Gamma" in dataset:
        x, y = Gamma_dataset.get_toy_dataset(num_samples=N, x=x)
        f = np.zeros(len(x))
    elif "Tukey" in dataset:
        x, y = Tukey_dataset.get_toy_dataset(num_samples=N)
        f = np.zeros(len(x))
    else:
        raise NameError("Undefined dataset!")
    return x, y, f

def get_error(obs, pred):
    """
    calculate eps = obs - pred
    """
    return obs - pred

def beta_calibration(b, x_train, y_train, dist, out_dir):
    """
    Given a beta-value and training data, learn var1 and var2 for the
    distribution.
    input:
        b: beta-value for ACCRUE loss function (b \in [0,1])
        x_train: input to NN
        y_train: pairs of obs, pred
        dist: error distribution the NN is assuming (current options AL or TPG)
    output:
        [crps, rs]: the CRPS and RS values for the training data of the
                    optimized NN
    """
    my_params = out_dir+"params"+str(b)+".pt"

    net_regr=None
    if dist=="AL":
        net_regr = NN_regression_AL.VarNet(
        module=NN_regression_AL.RegressorModule,
        beta=b,
        max_epochs=1000,
        optimizer=torch.optim.Adam,
        batch_size=100, 
        lr=5e-3,
        callbacks=[callbacks.EarlyStopping(monitor='valid_loss', patience=10),
                    Checkpoint(f_params=my_params)],
        train_split=skorch_dataset.ValidSplit(0.2),
        verbose=0
        )
    elif dist=="TPG":
        net_regr = NN_regression_TPG.VarNet(
        module=NN_regression_TPG.RegressorModule,
        beta=b,
        max_epochs=1000,
        optimizer=torch.optim.Adam,
        lr=5e-3,
        batch_size=100,
        callbacks=[callbacks.EarlyStopping(monitor='valid_loss', patience=10),
                    Checkpoint(f_params=my_params)],
        train_split=skorch_dataset.ValidSplit(0.2), # 2 would be train set is first 1/2 and valid second 1/2
        verbose=0
        )
    else:
        raise NameError("Undefined distribution!")

    net_regr.initialize_criterion()
    net_regr.criterion_

    # automatically splits x_train into 64% training and 36% validation data
    net_regr.fit(x_train, y_train)
    net_regr.load_params(f_params=my_params)

    pred_all = net_regr.predict(x_train)
    pred_all = np.exp(pred_all)
    # print(pred_all)
    pred_var1 = pred_all[:,0]
    pred_var2 = pred_all[:,1]
    # pred_lam=pred_all

    eps = y_train[:,0] - y_train[:,1]
    crps=-1
    rs=-1
    if dist=="AL":
        crps = asymmLaplace_accrue_torch.get_avg_CRPS_torch(
                            torch.tensor(pred_var1),
                            torch.tensor(eps), lam=torch.tensor(pred_var2))
        rs = asymmLaplace_accrue_torch.analytical_RS_torch(
                            torch.tensor(eps),
                            torch.tensor(pred_var1), torch.tensor(pred_var2))
    elif dist=="TPG":
        crps = twoPieceGauss_accrue_torch.get_avg_CRPS_torch(
                            torch.tensor(pred_var1),
                            torch.tensor(pred_var2), torch.tensor(eps))
        rs = twoPieceGauss_accrue_torch.analytical_RS_torch(
                            torch.tensor(eps),
                            torch.tensor(pred_var1), torch.tensor(pred_var2))
    # print("b:", b, crps, rs)
    return [crps, rs]

def get_optimal_beta(dist,x,y,f,out_dir):
    """
    Given calibration data, find the best beta-value (minimizes ACCRUE)
    input:
        dist: error distribution (AL or TPG)
        x, y, f: training input, obs, and predictions
        out_file: file to write resulting info into
    output:
        beta: optimal beta value for the calibration data
    """
    out_file = out_dir+"opt_beta.txt"

    # loop through beta values to select the "best" given calibration data
    betas = np.arange(0.1, 1.0, 0.1)
    crps_beta = np.zeros(len(betas))
    rs_beta = np.zeros(len(betas))
    y_train = np.vstack([y.T, f.T]).T
    y_train = y_train.astype(np.float32)
    x_train = np.reshape(x, (-1,1))
    x_train = x_train.astype(np.float32)

    with multiprocessing.Pool(len(betas)) as p:
        result= np.array(p.starmap(beta_calibration, zip(betas, repeat(x_train),
                                    repeat(y_train), repeat(dist), repeat(out_dir))))
        crps_beta=result[:,0]
        rs_beta=result[:,1]

    a = rs_beta**2
    b = crps_beta**2
    c = np.sqrt(a+b)
    ind = np.argsort(c)

    info = np.array([ind, betas, crps_beta, rs_beta])
    np.savetxt(out_file, info)
    return betas[ind[0]]

def get_best_net(dist,x_train, y_train, x_test, y_test, beta, out_dir, samp):
    """
    Generate 1 random NN, and return the one that minimizes the testing data's
    loss
    """
    best_net = None
    min_loss = np.Inf
    i = 0
    # gaurantee at least five starting config works
    while (i < 5) or (best_net == None and i >= 5):
        my_params=out_dir+"params"+str(samp)+"_"+str(i)+".pt"
        net_i=None
        if dist=="AL":
            net_i = NN_regression_AL.VarNet(
                module=NN_regression_AL.RegressorModule,
                beta=beta,
                max_epochs=1000,
                # optimizer=torch.optim.RMSprop,
                optimizer=torch.optim.Adam,
                lr=5e-3,
                batch_size=100,
                callbacks=[callbacks.EarlyStopping(monitor='valid_loss',
                           patience=10),
                           Checkpoint(f_params=my_params)],
                # need to save the best model not the last
                train_split=skorch_dataset.ValidSplit(0.2),
                verbose=0
            )
        elif dist=="TPG":
            net_i = NN_regression_TPG.VarNet(
                module=NN_regression_TPG.RegressorModule,
                beta=beta,
                max_epochs=1000,
                optimizer=torch.optim.Adam,
                lr=5e-3,
                batch_size=100,
                callbacks=[callbacks.EarlyStopping(monitor='valid_loss',
                           patience=10),
                           Checkpoint(f_params=my_params)],
                train_split=skorch_dataset.ValidSplit(0.2),
                verbose=0
            )

        net_i.initialize_criterion()
        net_i.criterion_

        net_i.fit(x_train, y_train)
        net_i.load_params(f_params=my_params)
        sd_i = net_i.predict(x_test)
        ar_i = net_i.get_loss(torch.from_numpy(sd_i), torch.from_numpy(y_test),
                                X=torch.from_numpy(x_test))
        ar_i = ar_i.detach().numpy()
        if not(np.isnan(ar_i)) and ar_i < min_loss:
            best_net = net_i
            min_loss = ar_i

        if ar_i != np.nan:
            i += 1

    return best_net, min_loss

def learn_dist(i, dist, x_interp, beta, N, out_dir, dataset="NO"):
    """
    Generate a replicate from the dataset and error dist and learn the
    parameters var1 and var2.
    Output:
        [var1(x_interp), var2(x_interp)]: the parameter outputs interpolated to
                                          the given inputs
    """
    np.random.seed(seed=i)
    random.seed(i)
    torch.manual_seed(i)

    N_train = int(N * 0.8)
    N_test = int(N * 0.2)
    
    x,y,f = get_data(dist, dataset=dataset, N=N)
    temp = np.array([x,y,f]).T

    # independent samples of x for train and test sets
    # validation set is included in train
    rands = temp[np.random.choice(temp.shape[0], N, replace=False), :]
    x_train = rands[0:N_train,0]
    x_test = rands[N_train:,0]
    ind_train = np.argsort(x_train)
    ind_test = np.argsort(x_test)
    x_train = x_train[ind_train]
    x_test = x_test[ind_test]

    # generate obs
    y_train = rands[0:N_train,1]
    y_test = rands[N_train:,1]
    y_train = y_train[ind_train]
    y_test = y_test[ind_test]

    pred_train = rands[0:N_train,2]
    pred_test = rands[N_train:,2]
    pred_train = pred_train[ind_train]
    pred_test = pred_test[ind_test]

    eps = get_error(y_train, pred_train)

    # reformatting for PyTorch
    y_train = np.vstack([y_train.T, pred_train.T]).T
    y_train = y_train.astype(np.float32)
    x_train = np.reshape(x_train, (-1,1))
    x_train = x_train.astype(np.float32)
    y_test = np.vstack([y_test.T, pred_test.T]).T
    y_test = y_test.astype(np.float32)
    x_test = np.reshape(x_test, (-1,1))
    x_test = x_test.astype(np.float32)
    
    net, test_loss = get_best_net(dist, x_train, y_train, x_test, y_test, beta,
                                  out_dir, i)

    with open(out_dir+"nets.pkl", 'ab') as f:
        pickle.dump(net, f)

    # calc parameters on TESTing data
    pred_all = np.exp(net.predict(x_test))
    pred_var1 = pred_all[:,0]
    pred_var2 = pred_all[:,1]

    # given pred_vars interpolate values for x_interp values
    func1 = interpolate.interp1d(x_test.flatten(), pred_var1.flatten(),
                    fill_value="extrapolate", assume_sorted=True)
    new_var1 = func1(x_interp)
    func2 = interpolate.interp1d(x_test.flatten(), pred_var2.flatten(),
                    fill_value="extrapolate", assume_sorted=True)
    new_var2 = func2(x_interp)
    
    return [new_var1, new_var2]

def generate_dist_params(dist, x_interp, beta, out_dir, samp=10, dataset="NO",
                         N=1000):
    """
    return samp samples of var1 and var2
    """
    var1_sum = np.zeros((samp, len(x_interp)))
    var2_sum = np.zeros((samp, len(x_interp)))
    all_nets=None

    # for i in range(samp):
    i = np.arange(samp)
    with multiprocessing.Pool(samp) as p:
        result = np.array(p.starmap(learn_dist, zip(i, repeat(dist),
                                    repeat(x_interp), repeat(beta), repeat(N),
                                    repeat(out_dir), repeat(dataset))))
        var1_sum=result[:,0]
        var2_sum=result[:,1]
    np.savetxt(out_dir+"inputs.txt", x_interp)
    np.savetxt(out_dir+"var1.txt", var1_sum)
    np.savetxt(out_dir+"var2.txt", var2_sum)
    return var1_sum, var2_sum

def loadall(filename):
    with open(filename, "rb") as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                break

if __name__ == "__main__":
    """
    Main driving function
    """
    # set random seed for reproducibility
    np.random.seed(seed=14)
    N = 1000
    dist="TPG"
    dataset="NO_linear_trig"

    # calibration data for determining beta
    x,y,f = get_data(dist, dataset=dataset, N=N)

    out_dir = "../output_"+dataset+"_"+dist+"/"

    beta=get_optimal_beta(dist,x,y,f,out_dir)
    # beta=0.9
    print("optimal beta = ", beta)
    
    x_interp = np.arange(0.0,1.001,0.001, dtype=float)

    # remove old file before generating new nets
    try:
        os.remove(out_dir+"nets.pkl")
    except FileNotFoundError:
        pass
    var1,var2 = generate_dist_params(dist, x_interp, beta, out_dir, samp=10,
                                     dataset=dataset, N=N)
