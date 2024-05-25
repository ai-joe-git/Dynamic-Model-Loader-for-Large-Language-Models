class DynamicModelInference:
    def __init__(self, model_loader):
        self.model_loader = model_loader

    def forward(self, input_ids):
        # Load embedding
        embedding = self.model_loader.load_embedding()
        x = embedding(input_ids)

        # Process each layer dynamically
        for idx in range(self.model_loader.num_layers):
            layer = self.model_loader.load_layer(idx)
            x = layer(x)
            self.model_loader.unload_layer(idx)  # Unload layer after use

        # Load and apply the output layer
        lm_head = self.model_loader.load_lm_head()
        logits = lm_head(x)
        
        return logits