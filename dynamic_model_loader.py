import torch
from transformers import AutoConfig, GPT2LMHeadModel

class DynamicModelLoader:
    def __init__(self, model_name, num_layers):
        self.config = AutoConfig.from_pretrained(model_name)
        self.num_layers = num_layers
        self.layers = [None] * num_layers
        self.embedding = None
        self.lm_head = None

    def load_layer(self, idx):
        if self.layers[idx] is None:
            layer = GPT2LMHeadModel(self.config).transformer.h[idx]
            layer.load_state_dict(torch.load(f'layer_{idx}.pt'))
            self.layers[idx] = layer
        return self.layers[idx]

    def unload_layer(self, idx):
        self.layers[idx] = None

    def load_embedding(self):
        if self.embedding is None:
            embedding = GPT2LMHeadModel(self.config).transformer.wte
            embedding.load_state_dict(torch.load('embedding.pt'))
            self.embedding = embedding
        return self.embedding

    def unload_embedding(self):
        self.embedding = None

    def load_lm_head(self):
        if self.lm_head is None:
            lm_head = GPT2LMHeadModel(self.config).lm_head
            lm_head.load_state_dict(torch.load('lm_head.pt'))
            self.lm_head = lm_head
        return self.lm_head

    def unload_lm_head(self):
        self.lm_head = None