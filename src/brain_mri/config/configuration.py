import os
import torch
# Directory of this config file
CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
# Navigate up 3 levels to reach the root directory from src/brain_mri/config/configuration.py
ROOT_DIR = os.path.abspath(os.path.join(CONFIG_DIR, "..", "..", ".."))
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']
DISPLAY_NAMES = {
    'glioma': 'Glioma',
    'meningioma': 'Meningioma',
    'notumor': 'No Tumor',
    'pituitary': 'Pituitary Tumor'
}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 224
# Look for model in artifacts/models/best_model.pth or fallback to root best_model.pth
MODEL_PATH = os.path.join(ROOT_DIR, "artifacts", "models", "best_model.pth")
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(ROOT_DIR, "best_model.pth")