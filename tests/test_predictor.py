import os
import sys
import unittest
import torch
import torch.nn as nn
from PIL import Image

# Ensure the src folder is in Python path for test execution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from brain_mri.inference.predictor import preprocess_image, predict, load_model_weights
from brain_mri.config.configuration import IMAGE_SIZE, CLASS_NAMES

class TestPredictor(unittest.TestCase):
    def setUp(self):
        # Create a dummy image for tests
        self.dummy_image = Image.new("RGB", (256, 256), color="white")

    def test_preprocess_image(self):
        device = torch.device("cpu")
        tensor = preprocess_image(self.dummy_image, IMAGE_SIZE, device)
        
        # Assert type
        self.assertIsInstance(tensor, torch.Tensor)
        # Assert dimensions: batch size 1, 3 channels, H=IMAGE_SIZE, W=IMAGE_SIZE
        self.assertEqual(tensor.shape, (1, 3, IMAGE_SIZE, IMAGE_SIZE))
        # Assert device
        self.assertEqual(tensor.device, device)

    def test_predict_mock(self):
        # Define a simple dummy neural network that outputs logit scores
        class DummyModel(nn.Module):
            def forward(self, x):
                # Output dummy logits for the 4 classes
                return torch.tensor([[1.0, 4.0, 2.0, 3.0]])

        model = DummyModel()
        tensor = torch.zeros((1, 3, IMAGE_SIZE, IMAGE_SIZE))
        
        pred_idx, confidence, probabilities = predict(model, tensor)
        
        # Verification
        # Index 1 should be predicted because logit 4.0 is the maximum
        self.assertEqual(pred_idx, 1)
        self.assertIsInstance(pred_idx, int)
        self.assertIsInstance(confidence, float)
        # Conf should be between 0 and 1
        self.assertTrue(0.0 <= confidence <= 1.0)
        # Probabilities should sum up to approximately 1.0
        self.assertLess(abs(probabilities.sum() - 1.0), 1e-5)
        self.assertEqual(len(probabilities), len(CLASS_NAMES))

    def test_load_model_weights_invalid_path(self):
        # Attempting to load from an invalid path should raise an error
        with self.assertRaises(FileNotFoundError):
            load_model_weights("non_existent_model_file.pth", torch.device("cpu"), CLASS_NAMES)

if __name__ == "__main__":
    unittest.main()
