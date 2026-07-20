import os
import sys
import unittest
from unittest.mock import patch
from pathlib import Path
import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image

# Ensure the src folder is in Python path for test execution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import brain_mri.inference.predictor as predictor
from brain_mri.inference.predictor import preprocess_image, predict, load_model_weights
from brain_mri.config.configuration import IMAGE_SIZE, CLASS_NAMES, DEVICE

class TestPredictor(unittest.TestCase):
    def setUp(self):
        # Create a dummy image for tests
        self.dummy_image = Image.new("RGB", (256, 256), color="white")

    def test_preprocess_image(self):
        """Preprocess should return a tensor with correct shape and type."""
        tensor = preprocess_image(self.dummy_image)
        
        # Assert type
        self.assertIsInstance(tensor, torch.Tensor)
        # Assert dimensions: batch size 1, 3 channels, H=IMAGE_SIZE, W=IMAGE_SIZE
        self.assertEqual(tensor.shape, (1, 3, IMAGE_SIZE, IMAGE_SIZE))
        # Assert device
        self.assertEqual(tensor.device, DEVICE)

    def test_predict_mock(self):
        """Predict should return a valid class, confidence, and probability distribution."""
        # Define a simple dummy neural network that outputs logit scores
        class DummyModel(nn.Module):
            def forward(self, x):
                # Output dummy logits for the 4 classes
                return torch.tensor([[1.0, 4.0, 2.0, 3.0]], device=x.device)

        model = DummyModel()
        tensor = torch.zeros((1, 3, IMAGE_SIZE, IMAGE_SIZE))
        
        result = predict(model, tensor)
        pred_idx = result["class_index"]
        pred_name = result["class_name"]
        confidence = result["confidence"]
        probabilities = result["probabilities"]
        
        # Verification
        # Index 1 should be predicted because logit 4.0 is the maximum
        self.assertEqual(pred_idx, 1)
        self.assertEqual(pred_name, CLASS_NAMES[1])
        self.assertIsInstance(pred_idx, int)
        self.assertIsInstance(confidence, float)
        # Conf should be between 0 and 1
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
        # Probabilities should sum up to approximately 1.0
        self.assertAlmostEqual(probabilities.sum(), 1.0, places=5)
        self.assertEqual(len(probabilities), len(CLASS_NAMES))

    def test_load_model_weights_invalid_path(self):
        """Verify that loading weights from a non-existent path raises FileNotFoundError."""
        with patch.object(predictor, "MODEL_PATH", Path("non_existent_model_file.pth")):
            with self.assertRaises(FileNotFoundError):
                load_model_weights()

    def test_model_eval_mode(self):
        """Verify that the loaded model is set to evaluation mode."""
        # Dynamically generate a valid state dict so we don't need to patch load_state_dict
        model_fn = getattr(models, predictor.MODEL_NAME)
        temp_model = model_fn(weights=None)
        num_features = temp_model.classifier[1].in_features
        temp_model.classifier[1] = nn.Linear(num_features, len(CLASS_NAMES))
        valid_state_dict = temp_model.state_dict()

        with patch("brain_mri.inference.predictor.torch.load", return_value=valid_state_dict):
            model = load_model_weights()
            self.assertFalse(model.training)

    def test_invalid_architecture(self):
        """Verify that an invalid architecture configuration raises a ValueError."""
        with patch.object(predictor, "MODEL_NAME", "resnet50"):
            with self.assertRaises(ValueError):
                load_model_weights()


if __name__ == "__main__":
    unittest.main()
