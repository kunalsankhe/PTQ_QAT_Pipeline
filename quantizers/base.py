import torch

class FakeQuantizeSTE(torch.autograd.Function):
  @staticmethod
  def forward(ctx, inputs, scale, zero_point, quant_min, quant_max):
    #float to fixed-point conversion
    quantized = torch.round(inputs / scale + zero_point)
    #clamp to hardware bit-width limits
    quantized = torch.clamp(quantized, quant_min, quant_max)
    # dequantize 
    dequantized = (quantized - zero_point) * scale 
    return dequantized

  @staticmethod
  def backward(ctx, grad_output):
    # straight-through estimator: pass gradients unaltered
    return grad_output, None, None, None, None

def fake_quantize(inputs, scale, zero_point, quant_min, quant_max):
  return FakeQuantizeSTE.apply(inputs, scale, zero_point, quant_min, quant_max)
