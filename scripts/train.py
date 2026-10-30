import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from quantizers.layers import QuantizedLinear

# 1. Define a simple image classification network
class SimpleQATNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = QuantizedLinear(28 * 28, 128, num_bits=8)
        self.relu = nn.ReLU()
        self.fc2 = QuantizedLinear(128, 10, num_bits=8)

    def set_quantization_mode(self, mode):
        self.fc1.set_quantization_mode(mode)
        self.fc2.set_quantization_mode(mode)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load standard MNIST dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    model = SimpleQATNet().to(device)
    criterion = nn.CrossEntropyLoss()
    
    # Phase A: Standard FP32 Pre-training (Simulated briefly here)
    print("--- Phase A: FP32 Pre-training ---")
    model.set_quantization_mode('ptq_calibrate') # Acts as pass-through for FP32
    optimizer_fp32 = optim.Adam(model.parameters(), lr=1e-3)
    
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer_fp32.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer_fp32.step()
        if batch_idx == 100: # Shortened for demonstration
            print(f"FP32 Loss: {loss.item():.4f}")
            break

    # Phase B: PTQ Calibration
    print("\n--- Phase B: PTQ Calibration ---")
    model.eval() # Disable dropout/batchnorm updates
    model.set_quantization_mode('ptq_calibrate')
    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            _ = model(data)
            if batch_idx == 50: # Calibrate on 50 batches
                break
    print("Calibration complete. Scales and Zero Points locked.")

    # Phase C: QAT Fine-tuning
    print("\n--- Phase C: QAT Fine-tuning ---")
    model.train()
    model.set_quantization_mode('qat_train')
    
    # Crucial QAT Step: Lower the learning rate significantly (e.g., 1e-5)
    optimizer_qat = optim.Adam(model.parameters(), lr=1e-5)
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer_qat.zero_grad()
        
        # Forward pass now injects quantization noise. 
        # Backward pass uses the Straight-Through Estimator.
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer_qat.step()
        
        if batch_idx % 20 == 0:
            print(f"QAT Batch {batch_idx} Loss: {loss.item():.4f}")
        if batch_idx == 100:
            break
            
    print("\nPhase 1 Complete. The model is now quantization-aware.")

if __name__ == "__main__":
    main()
