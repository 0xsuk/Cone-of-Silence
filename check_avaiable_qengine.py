import torch
print("Available quantized engines:", torch.backends.quantized.supported_engines)
print("Current engine:", torch.backends.quantized.engine)
