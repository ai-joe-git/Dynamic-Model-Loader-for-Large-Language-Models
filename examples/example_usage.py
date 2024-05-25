import torch
from dynamic_model_loader import DynamicModelLoader
from dynamic_model_inference.py import DynamicModelInference

# Example usage
model_loader = DynamicModelLoader("big-model", num_layers=24)
dynamic_model = DynamicModelInference(model_loader)
input_ids = torch.tensor([[1, 2, 3]])
outputs = dynamic_model.forward(input_ids)
print(outputs)