import torch

# Path to your .pt file
path = "D:\Skills\new\NeurIPS2\outputs\torch_tensor\selfies.pt"  # Change if the path is different

# Load the saved file (dict or tensor)
data = torch.load(path)

# Inspect keys (if saved as a dict)
if isinstance(data, dict):
    print("Keys in .pt file:", list(data.keys()))
    for key, val in data.items():
        print(f"\n--- {key} ---")
        print("Type:", type(val))
        if isinstance(val, torch.Tensor):
            print("Shape:", val.shape)
            print("Dtype:", val.dtype)
            print("Sample:\n", val[:5])  # Show first 5 rows
        elif isinstance(val, list):
            print("Sample:\n", val[:5])
        elif isinstance(val, dict):
            print("Keys:", list(val.keys()))
else:
    # If just a tensor
    print("Loaded tensor:")
    print("Shape:", data.shape)
    print("Dtype:", data.dtype)
    print("Sample:\n", data[:5])
