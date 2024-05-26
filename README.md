# Dynamic Model Loader for Large Language Models

**Project Goal:** 
Enable access to large language models (LLMs) for individuals with limited hardware resources by dynamically loading and unloading model parts to fit within low RAM environments. This project aims to bridge the technology gap for those in third-world countries.

## Overview

This repository provides a framework for running large models on devices with limited memory by splitting the model into smaller chunks and dynamically loading and unloading these chunks during inference. This approach allows for the utilization of very large models, such as 70B parameter models, on CPUs with low RAM.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- PyTorch
- Transformers library from Hugging Face

### Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/ai-joe-git/Dynamic-Model-Loader-for-Large-Language-Models.git
    cd dynamic-model-loader
    ```

2. **Install the required packages:**

    ```bash
    pip install torch transformers
    ```

### Usage

#### 1. Segment the Model

First, segment your large model into smaller parts. This example uses a transformer-based model:

```python
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
```

#### 2. Dynamic Loading and Unloading

Create a class to manage the loading and unloading of these model segments:

```python
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
```

#### 3. Integrate with Inference

Implement a mechanism to handle inference by dynamically managing the model segments:

```python
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

# Example usage
model_loader = DynamicModelLoader("big-model", num_layers=24)
dynamic_model = DynamicModelInference(model_loader)
input_ids = torch.tensor([[1, 2, 3]])
outputs = dynamic_model.forward(input_ids)
```

### Contributions

We welcome contributions from the community to improve this project. Here are some ways you can help:

- **Enhance the Loading Mechanism:** Improve the efficiency of loading and unloading model parts.
- **Add Caching:** Implement a caching mechanism to reduce disk I/O.
- **Asynchronous Loading:** Use asynchronous I/O operations to preload layers while others are being processed.
- **Documentation:** Improve documentation and add more usage examples.
- **Testing:** Write tests to ensure the functionality and reliability of the dynamic loading mechanism.
- **Optimization for Specific Hardware:** Adapt the code to better utilize specific hardware configurations common in low-resource environments.

### How to Contribute

1. **Fork the Project:**
   Click the "Fork" button at the top right of this page to create a copy of this repository under your own GitHub account.

2. **Clone Your Fork:**
   ```bash
   git clone https://github.com/ai-joe-git/Dynamic-Model-Loader-for-Large-Language-Models.git
   cd dynamic-model-loader
   ```

3. **Create a Branch:**
   Create a branch for your feature, enhancement, or bug fix.
   ```bash
   git checkout -b your-feature-branch
   ```

4. **Make Your Changes:**
   Make your changes to the codebase or documentation.

5. **Commit Your Changes:**
   Commit your changes with a meaningful commit message.
   ```bash
   git add .
   git commit -m "Add feature X or Fix bug Y"
   ```

6. **Push to Your Fork:**
   Push your changes to your forked repository.
   ```bash
   git push origin your-feature-branch
   ```

7. **Create a Pull Request:**
   Go to the original repository and click "New Pull Request" from your branch.

### License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

### Acknowledgments

- [Hugging Face](https://huggingface.co/) for the Transformers library.
- The open-source community for continuous support and contributions.

### Contact

For any questions or suggestions, feel free to open an issue or contact the repository owner directly.

---

Together, we can make advanced AI accessible to everyone, regardless of their hardware limitations. Let's bridge the technology gap and empower more people with the tools they need to succeed!

```
