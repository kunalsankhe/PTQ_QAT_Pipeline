import torch
import torch.nn as nn
import torch.nn.functional as F
from .min_max import MinMaxQuantizer

class QuantizedLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, num_bits=8):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # The underlying FP32 parameters that will be updated by QAT gradients
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
            
        # Initialize weights (standard PyTorch Kaiming initialization)
        nn.init.kaiming_uniform_(self.weight, a=torch.math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / torch.math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

        # 1. Weight Quantizer (Symmetric: centered around 0)
        self.weight_quantizer = MinMaxQuantizer(num_bits=num_bits, symmetric=True)
        
        # 2. Activation Quantizer (Asymmetric: handles shifted distributions like ReLU)
        self.activation_quantizer = MinMaxQuantizer(num_bits=num_bits, symmetric=False)

    def set_quantization_mode(self, mode):
        # Helper to easily switch the layer between 'ptq_calibrate', 'qat_train', and 'inference'
        self.weight_quantizer.mode = mode
        self.activation_quantizer.mode = mode

    def forward(self, x):
        # 1. Quantize the incoming activations
        q_x = self.activation_quantizer(x)
        
        # 2. Quantize the weights
        q_weight = self.weight_quantizer(self.weight)
        
        # 3. Execute the linear operation using the pseudo-quantized tensors
        # (The bias is typically kept in higher precision like FP32 or INT32 in hardware, 
        # so we pass it through unquantized here for standard simulation)
        return F.linear(q_x, q_weight, self.bias)
