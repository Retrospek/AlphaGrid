# ~~~~~~~~~~~~~~~~~~~~
## MAYBE IN THE FUTURE I'LL FINISH THIS FILE LOL
# ~~~~~~~~~~~~~~~~~~~~

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"

import numpy as np
import math
import random

from datetime import datetime

class GARCH(nn.Module):
    def __init__(self, p, q):
        super().__init__()

        self.p = p
        self.q = q

        alphas = [nn.Parameter(torch.tensor(0.1))]
        betas = []

        for p_idx in range(p):
            alphas.append(nn.Parameter(torch.tensor(0.2)))
        for q_idx in range(q):
            betas.append(nn.Parameter(torch.tensor(0.2)))
        
        self.alphas = nn.ParameterList(alphas)
        self.betas = nn.ParameterList(betas)

    def forward(self, a_prevs, sigma_prevs):
        alpha0_tensor = torch.tensor([1.0], dtype=a_prevs.dtype, device=a_prevs.device)
        a_prevs = torch.cat([alpha0_tensor, a_prevs])

        a_prevs_squared = torch.square(a_prevs)
        sigma_prevs_squared = torch.square(sigma_prevs)

        alpha_weights = torch.stack([param for param in self.alphas])  # shape: (p+1,)
        beta_weights  = torch.stack([param for param in self.betas])   # shape: (q,)

        alpha_results = torch.dot(a_prevs_squared, alpha_weights)
        beta_results = torch.dot(sigma_prevs_squared, beta_weights)

        epsilon = torch.randn(1)

        result = epsilon * torch.sqrt(alpha_results + beta_results)
        
        return result
    
    def fit(self, dataloader, epoch_num, criterion, LR=1e-3):

        self.dataloader = dataloader
        self.epoch_num = epoch_num
        self.learning_rate = LR
        self.criterion = criterion
        self.optimizer = optim.Adam(params=self.parameters(), lr=LR)

        self.epoch_losses = []
        self.batch_losses = []

        for epoch in range(epoch_num):
            epoch_loss = 0.0
            for a_prevs, sigma_prevs, target_a_curr in dataloader:
                a_prevs, sigma_prevs, target_a_curr = a_prevs.to(device), sigma_prevs.to(device), target_a_curr.to(device)
                
                self.optimizer.zero_grad()
                res = self(a_prevs=a_prevs, sigma_prevs=sigma_prevs)

                loss = self.criterion(res, target_a_curr)

                epoch_loss += loss.item()
                self.batch_losses.append(loss.item())

                loss.backward()
                self.optimizer.step()

            print(f"Epoch Loss for Garch({self.p}, {self.q}) ==> {epoch_loss}")
            self.epoch_losses.append(epoch_loss)