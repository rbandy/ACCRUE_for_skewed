import numpy as np
# import scipy.stats
# from scipy.stats import laplace_asymmetric
import torch

def CRPS_asymmLaplace_ana_torch(kappa, error, lam=1.0):
    """
    continuous rank probability score
    assuming asymmetric Laplace distribution AL(0, lambda, kappa)
    input: kappa = asymmetry parameter
           error = y_0 - pred
           error between a given obs, y_0, and corresponding prediction, pred.
           lambda = scale
    """
    # kappa=torch.tensor(kappa)
    lam=torch.div(1,lam)
    assert kappa > 0 and lam > 0
    # 1+kappa^2
    const = torch.add(1, torch.pow(kappa,2))
    # 2*lam*(1+kappa^2)^2
    const2 = torch.mul(torch.mul(2,lam), torch.pow(const,2))
    
    if error <= 0:
        # (2*kappa^3)/(lam*(1+kappa^2))
        const1 = torch.div(torch.mul(2, torch.pow(kappa,3)), 
                          (torch.mul(lam, const)))
        # print("after const1 error <0")
        # |error| + const1*exp(lam/kappa*error) - const1 + kappa^5/(const2) + 1/(const2*kappa)
        return torch.add(torch.add(torch.sub(torch.add(torch.abs(error), 
                         torch.mul(const1, torch.exp(torch.mul(torch.div(lam, kappa), error)))),
                        const1),
                        torch.div(torch.pow(kappa, 5), const2)),
                        torch.div(1, torch.mul(const2, kappa)))

    # error > 0 
    # 2/(lam*kappa*(1+kappa^2))
    const1 = torch.div(2, torch.mul(torch.mul(lam, kappa), const))
    # print("after const1 error >0")
    # |error| + const1*exp(-lam*kappa*error) - const1 + kappa^5/const2 + 1/(kappa*const2)
    return torch.add(torch.add(torch.sub(torch.add(torch.abs(error), 
                         torch.mul(const1, torch.exp(torch.mul(torch.mul(-lam, kappa), error)))),
                        const1),
                        torch.div(torch.pow(kappa, 5), const2)),
                        torch.div(1, torch.mul(const2, kappa)))


def get_avg_CRPS_torch(kappa, error, lam=None):
    """
    compute the mean CRPS over many observations
    inputs: kappa = array of asymm params
            error = array of errors
            len(kappa) == len(error)
            optional lam = array of scale params len(lam) == len(error)
    """
    N = kappa.shape[0]
    crps = torch.zeros(N)
    if lam is None:
        lam = [1.0]*len(error)
    for i in range(len(error)):
        eps = error[i]
        k = kappa[i]
        l = lam[i]
        crps[i] = CRPS_asymmLaplace_ana_torch(k, eps, lam=l)
        # print("crps ", i, crps[i])
        
    return torch.div(torch.sum(crps), N)

def calc_eta_torch(error, kappa, lam):
    eta = torch.zeros(len(error))
    lam=torch.div(1,lam)
    for k in range(len(error)):
        k2 = torch.pow(kappa[k], 2)
        if error[k] <= 0:
            eta[k] = torch.mul(torch.div(k2, (1+k2)), 
                               torch.exp(torch.mul(torch.div(lam[k], kappa[k]), 
                                                   error[k])))
        else:
            eta[k] = torch.sub(1, torch.mul(torch.div(1, (1+k2)), 
                               torch.exp(torch.mul(torch.mul(-lam[k], kappa[k]), 
                                                   error[k]))))
    return torch.sort(eta)

def analytical_RS_torch(error, kappa, lam):
    """
    inputs: 
        emp_eta: sorted array of CDF transforms 
                 AL.CDF(m=error, kappa=curr_kappa, lambda=curr_lam)
    output: removing -2/3 constant!!
    1/N * \sum_{i=1}^N (emp_eta[i]^2) 
    + 1/N^2 *  \sum_{i=1}^N (i^2 * (eta[i+1]-eta[i]))
    """
    eta = torch.zeros(len(error))
    # kappa=torch.tensor(kappa)
    lam=torch.div(1,lam)
    for k in range(len(error)):
        k2 = torch.pow(kappa[k], 2)
        if error[k] <= 0:
            eta[k] = torch.mul(torch.div(k2, (1+k2)), 
                               torch.exp(torch.mul(torch.div(lam[k], kappa[k]), 
                                                   error[k])))
        else:
            eta[k] = torch.sub(1, torch.mul(torch.div(1, (1+k2)), 
                               torch.exp(torch.mul(torch.mul(-lam[k], kappa[k]), 
                                                   error[k]))))
    emp_eta = torch.sort(eta).values
    # print("sorted eta: ", emp_eta)

    N = len(emp_eta)
    temp = 0.0
    for i in range(N-1):
        temp += torch.mul((i+1)**2, torch.sub(emp_eta[i+1],emp_eta[i]))
    temp += torch.mul((N)**2, torch.sub(1, emp_eta[-1]))
    return torch.add(torch.div(torch.sum(torch.pow(emp_eta, 2)), N), torch.div(temp, (N**2)))

