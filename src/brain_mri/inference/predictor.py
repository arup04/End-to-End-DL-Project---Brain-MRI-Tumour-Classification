import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms

from brain_mri.config.configuration import (
    MODEL_NAME,
    MODEL_PATH,
    IMAGE_SIZE,
    DEVICE,
    CLASS_NAMES,
)

# Create transform once
TRANSFORM = transforms.Compose([
    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=transforms.InterpolationMode.LANCZOS,
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def load_model_weights():
    """
    Build the EfficientNet model specified in model.yaml,
    load trained weights, and return the model in evaluation mode.
    """

    # Ensure only EfficientNet architectures are allowed
    if not MODEL_NAME.startswith("efficientnet_"):
        raise ValueError(
            f"Unsupported architecture '{MODEL_NAME}'. "
            "Only EfficientNet models are currently supported."
        )

    # Dynamically load EfficientNet architecture
    model_fn = getattr(models, MODEL_NAME, None)

    if model_fn is None:
        raise ValueError(
            f"Architecture '{MODEL_NAME}' is not available in torchvision.models."
        )

    model = model_fn(weights=None)

    # Replace classification head
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, len(CLASS_NAMES))

    # Load trained weights
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)

    model.to(DEVICE)
    model.eval()

    return model


def preprocess_image(image):
    """
    Preprocess a PIL image for inference.
    """

    image_tensor = TRANSFORM(image)
    image_tensor = image_tensor.unsqueeze(0)

    return image_tensor.to(DEVICE)


def predict(model, image_tensor):
    """
    Perform inference on a preprocessed image.

    Returns:
        dict containing prediction details.
    """

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, prediction = torch.max(probabilities, dim=1)

    prediction = prediction.item()

    return {
        "class_index": prediction,
        "class_name": CLASS_NAMES[prediction],
        "confidence": confidence.item(),
        "probabilities": probabilities.squeeze().cpu().numpy(),
    }
