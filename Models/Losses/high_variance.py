import torch
import torch.nn as nn

class VolatilitySpikeLoss(nn.Module):
    def __init__(self, tau=2, spike_threshold=1.0, direction_weight=2):
        super().__init__()
        self.tau = tau
        self.spike_threshold = spike_threshold
        self.direction_weight = direction_weight
    
    def forward(self, y, y_hat):
        error = y - y_hat
        
        asymmetric_loss = torch.where(error > 0, (self.tau ** 2) * error, (self.tau) * error)
        
        # Weight HIGH volatility more heavily
        spike_weights = torch.where(y > self.spike_threshold, 10.0, 5.0)
        
        volatility_loss = torch.mean(spike_weights * asymmetric_loss.abs())
        
        # Sign matching penalty - penalize when signs don't match
        sign_mismatch = (torch.sign(y) != torch.sign(y_hat)).float()
        direction_penalty = torch.mean(sign_mismatch)
        
        return volatility_loss + self.direction_weight * direction_penalty
    
class ExponentialDistanceWeightedLoss(nn.Module):
    def __init__(self, scale=1.5):
        super().__init__()
        self.scale = scale
    
    def forward(self, y, y_hat):
        error = torch.abs(y - y_hat)
        
        # Exponential weighting - errors at |y|=2 get much higher penalty than |y|=0.5
        distance_weight = torch.exp(self.scale * torch.abs(y))
        
        return torch.mean(distance_weight * error)