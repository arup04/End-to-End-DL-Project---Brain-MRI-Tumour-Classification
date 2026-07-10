import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms

def load_model_weights(model_path, device, class_names):
    """
    Instantiate the EfficientNet-B2 model, modify the classification head,
    and load the model weights from model_path.
    """
    model = models.efficientnet_b2(weights=None)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, len(class_names))

    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def preprocess_image(image, image_size, device):
    """
    Preprocess PIL Image for EfficientNet model.
    """
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0).to(device)

def predict(model, tensor):
    """
    Run model forward pass and return predicted index, confidence score, and raw probabilities.
    """
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)
        conf, pred = torch.max(probs, 0)
    return pred.item(), conf.item(), probs.cpu().numpy()
