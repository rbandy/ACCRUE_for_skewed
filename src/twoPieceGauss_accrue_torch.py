import numpy as np
import torch

def norm_CDF(x, sd):
    temp = torch.erf(torch.div(x, torch.mul(np.sqrt(2), sd)))
    return torch.add(0.5, torch.mul(0.5, temp))

def norm_PDF(x, sd):
    coeff = torch.div(1, torch.mul(np.sqrt(2*torch.pi), sd))
    # coeff = 1/(np.sqrt(2*torch.pi))
    ex = torch.exp(torch.mul(-0.5, torch.pow(torch.div(x,sd), 2)))
    return torch.mul(coeff, ex)

def CRPS_TPG_ana_torch(sigma1, sigma2, error):
    """
    continuous rank probability score
    assuming two-piece Gaussian distribution SG(mu=0, sigma1, sigma2)
    input: sigma1 = sd of left side,
           sigma2 = sd of right side,
           error = y_0 - pred
           error between a given obs, y_0, and corresponding prediction, pred.
    """
    s1_2 = torch.pow(sigma1, 2)
    s2_2 = torch.pow(sigma2, 2)
    s1_3 = torch.pow(sigma1, 3)
    s2_3 = torch.pow(sigma2, 3)
    sum_s = torch.add(sigma1, sigma2)
    const = torch.div(torch.mul(4, s1_2), sum_s)
    const2 = torch.div(torch.mul(4, s2_2), sum_s)
    coeff = 2/(np.sqrt(torch.pi))
    den = torch.pow(sum_s, 2)

    
    if error <= 0:
        # const*[error/sigma1 * CDF(error, sigma1) + PDF(error, sigma1)]
        # - error
        # + coeff*[(sqrt{2}*sigma2*(s2_2-s1_2) - (s1_3+s2_3))/(den)]
        var = torch.div(error, sigma1)
        p1 = torch.mul(const, 
                        torch.add(torch.mul(torch.div(error, sigma1),
                                            norm_CDF(var, 1)),
                                    norm_PDF(var, 1)))
        p3 = torch.mul(coeff, 
                        torch.div(torch.sub(torch.mul(torch.mul(np.sqrt(2), sigma2), 
                                                        torch.sub(s2_2, s1_2)), 
                                            torch.add(s1_3, s2_3)),
                                    den))
        # print("case 1: ", p1, error, p3)
        return torch.add(torch.sub(p1, error), p3)
    # const*[error/sigma2 * CDF(error, sigma2) + PDF(error, sigma2)]
    # + [(den - 4*s2_2)/(den)]*error
    # + coeff*[(sqrt{2}sigma1*(s1_2 - s2_2) - (s1_3 + s2_3))/den]
    var = torch.div(error, sigma2)
    p1 = torch.mul(const2, 
                    torch.add(torch.mul(torch.div(error, sigma2),
                                        norm_CDF(var, 1)),
                                norm_PDF(var,1)))
    p2 = torch.mul(torch.div(torch.sub(torch.pow(torch.sub(sigma1, sigma2),2),
                    torch.mul(4, s2_2)), den), error)
    p3 = torch.mul(coeff, 
                    torch.div(torch.sub(torch.mul(torch.mul(np.sqrt(2), sigma1), 
                                                    torch.sub(s1_2, s2_2)), 
                                        torch.add(s1_3, s2_3)),
                                den))
    # print("case 2: ", p1, p2, p3)
    return torch.add(torch.add(p1, p2), p3)

    

def get_avg_CRPS_torch(sigma1, sigma2, error):
    """
    compute the mean CRPS over many observations
    inputs: sigma1 = array of left std
            sigma2 = array of right std
            error = array of errors
    """
    N = sigma1.shape[0]
    crps = torch.zeros(N)
    for i in range(len(error)):
        eps = error[i]
        s1 = sigma1[i]
        s2 = sigma2[i]
        crps[i] = CRPS_TPG_ana_torch(s1, s2, eps)
        # print("i", i, "CRPS", crps[i])
        if crps[i] < 0:
            print("NOOO!", eps, s1, s2, crps[i])
            break
        
    return torch.div(torch.sum(crps), N)

def calc_eta_torch(error, sigma1, sigma2):
    eta = torch.zeros(len(error))
    for k in range(len(error)):
        sum_s = torch.add(sigma1[k], sigma2[k])
        const = torch.div(sigma1[k], sum_s)
        if error[k] <= 0:
            eta[k] = torch.mul(const, torch.add(1, 
                            torch.special.erf(torch.div(error[k], sigma1[k]))))
        else:
            const2 = torch.div(sigma2[k], sum_s)
            eta[k] = torch.add(const, torch.mul(const2,
                            torch.special.erf(torch.div(error[k], sigma2[k]))))
    return torch.sort(eta).values

def analytical_RS_torch(error, sigma1, sigma2):
    """
    inputs: 
        emp_eta: sorted array of CDF transforms 
                 TPG.CDF(m=error, sigma1, sigma2)
    output:
    1/N * \\sum_{i=1}^N (emp_eta[i]^2) 
    + 1/N^2 *  \\sum_{i=1}^N (i^2 * (emp_eta[i+1]-emp_eta[i]))
    """
    emp_eta = calc_eta_torch(error, sigma1, sigma2)

    N = len(emp_eta)
    temp = 0.0
    for i in range(N-1):
        temp += torch.mul((i+1)**2, torch.sub(emp_eta[i+1],emp_eta[i]))
    temp += torch.mul((N)**2, torch.sub(1, emp_eta[-1]))
    return torch.add(torch.div(torch.sum(torch.pow(emp_eta, 2)), N), torch.div(temp, (N**2)))

