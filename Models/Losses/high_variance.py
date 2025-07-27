import torch
import torch.nn as nn

class VolatilitySpikeLoss(nn.Module):
    def __init__(self, tau=2, spike_threshold=1.0):
        super().__init__()
        self.tau = tau
        self.spike_threshold = spike_threshold
    
    def forward(self, y, y_hat):
        error = y - y_hat
        
        # Quantile loss
        quantile_loss = torch.where(error > 0, (self.tau ** 2) * error, (self.tau) * error)
        
        # Weight HIGH volatility more heavily
        spike_weights = torch.where(y > self.spike_threshold, 10, 5.0)
        
        return torch.mean(spike_weights * quantile_loss.abs())
    
class ExponentialDistanceWeightedLoss(nn.Module):
    def __init__(self, scale=1.5):
        super().__init__()
        self.scale = scale
    
    def forward(self, y, y_hat):
        error = torch.abs(y - y_hat)
        
        # Exponential weighting - errors at |y|=2 get much higher penalty than |y|=0.5
        distance_weight = torch.exp(self.scale * torch.abs(y))
        
        return torch.mean(distance_weight * error)