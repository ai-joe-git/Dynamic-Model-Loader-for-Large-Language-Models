import unittest
from dynamic_model_loader import DynamicModelLoader

class TestDynamicModelLoader(unittest.TestCase):
    def setUp(self):
        self.model_loader = DynamicModelLoader("big-model", num_layers=24)

def test_load_layer(self):
        layer = self.model_loader.load_layer(0)
        self.assertIsNotNone(layer, "Layer 0 should be loaded")
        self.model_loader.unload_layer(0)
        self.assertIsNone(self.model_loader.layers[0], "Layer 0 should be unloaded")

    def test_load_embedding(self):
        embedding = self.model_loader.load_embedding()
        self.assertIsNotNone(embedding, "Embedding should be loaded")
        self.model_loader.unload_embedding()
        self.assertIsNone(self.model_loader.embedding, "Embedding should be unloaded")

    def test_load_lm_head(self):
        lm_head = self.model_loader.load_lm_head()
        self.assertIsNotNone(lm_head, "LM Head should be loaded")
        self.model_loader.unload_lm_head()
        self.assertIsNone(self.model_loader.lm_head, "LM Head should be unloaded")

if __name__ == '__main__':
    unittest.main()