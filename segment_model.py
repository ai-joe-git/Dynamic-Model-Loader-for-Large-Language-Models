import torch
from transformers import AutoModelForCausalLM

# Load your large model
model = AutoModelForCausalLM.from_pretrained("big-model")

# Save each layer separately
for idx, layer in enumerate(model.transformer.h):
    torch.save(layer.state_dict(), f'layer_{idx}.pt')

# Save the embedding and output layers separately
torch.save(model.transformer.wte.state_dict(), 'embedding.pt')
torch.save(model.lm_head.state_dict(), 'lm_head.pt')