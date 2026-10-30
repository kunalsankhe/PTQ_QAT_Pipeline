import torch
import torch.nn as nn
from .base import fake_quantize

class MinMaxQuantizer(nn.Module):
    # Ensure "symmetric" is spelled correctly here
    def __init__(self, num_bits=8, symmetric=True):
        super().__init__()
        self.num_bits = num_bits
        self.symmetric = symmetric
        
        if symmetric:
            self.qmax = (2 ** (num_bits - 1)) - 1
            self.qmin = -self.qmax - 1
        else:
            self.qmax = (2 ** num_bits) - 1
            self.qmin = 0

        self.register_buffer('scale', torch.tensor(1.0))
        self.register_buffer('zero_point', torch.tensor(0.0))
        self.register_buffer('running_min', torch.tensor(float('inf')))
        self.register_buffer('running_max', torch.tensor(float('-inf')))
        
        self.mode = 'ptq_calibrate'

    def update_calibration_stats(self, x):
        current_min = x.detach().min()
        current_max = x.detach().max()
        self.running_min = torch.min(self.running_min, current_min)
        self.running_max = torch.max(self.running_max, current_max)
        
        if self.symmetric:
            max_abs = torch.max(torch.abs(self.running_min), torch.abs(self.running_max))
            max_abs = torch.clamp(max_abs, min=1e-5)
            self.scale.copy_(max_abs / self.qmax)
            self.zero_point.copy_(torch.tensor(0.0, device=x.device))
        else:
            range_val = torch.clamp(self.running_max - self.running_min, min=1e-5)
            self.scale.copy_(range_val / (self.qmax - self.qmin))
            zp = torch.round(self.qmin - (self.running_min / self.scale))
            self.zero_point.copy_(torch.clamp(zp, self.qmin, self.qmax))

    def forward(self, x):
        if self.mode in ['ptq_calibrate', 'qat_train']:
            self.update_calibration_stats(x)
            
        if self.mode == 'ptq_calibrate':
            return x
            
        return fake_quantize(x, self.scale, self.zero_point, self.qmin, self.qmax)
