import twoPieceGauss_accrue_torch

from skorch import NeuralNetRegressor
from skorch import NeuralNet
from skorch import callbacks
import torch
from torch import nn
import numpy as np

# construct the module aka NN architecture 
class RegressorModule(nn.Module):
    def __init__(self):
        super(RegressorModule, self).__init__()
        self.flatten = nn.Flatten() # flatten inputs
        # NN arch
        self.linear_relu_stack = nn.Sequential(
            nn.BatchNorm1d(1),
            nn.Linear(1, 10), # input dim currently 1
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.LeakyReLU(),
            nn.Linear(10, 2),
        )

    def forward(self, x, **kwargs):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        # output should be log(kappa)!
        return logits


# construct the NN with ACCRUE as the loss function
class VarNet(NeuralNetRegressor):
    def __init__(self, *args, beta=1.0, **kwargs):
        # save_last=False, 
        super().__init__(*args, **kwargs)
        self.beta = beta


    def initialize_criterion(self):
        self.criterion_ = self.get_loss
        return self
    

    def get_loss(self, y_pred, y_true, X=None, training=False):
        """
        loss = ACCRUE with AL expected distribution
        """
        # here y_pred will be predicted kappa values from the NN 
        # not output predictions 
        out = torch.exp(y_pred).squeeze() # guarantee sigma1, sigma2 >= 0
        sigma1 = out[:,0]
        sigma2 = out[:,1]

        # y_true[:,0] observations
        # y_true[:,1] model preditions
        eps = torch.subtract(y_true[:,0], y_true[:,1])

        mean_CRPS = twoPieceGauss_accrue_torch.get_avg_CRPS_torch(sigma1, sigma2, eps)

        RS = twoPieceGauss_accrue_torch.analytical_RS_torch(eps, sigma1, sigma2)

        return torch.add(torch.mul(self.beta, mean_CRPS), torch.mul((1-self.beta),RS))