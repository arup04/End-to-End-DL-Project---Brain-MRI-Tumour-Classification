from pathlib import Path
import torch
import yaml

# Resolve root directory
CONFIG_DIR = Path(__file__).resolve().parent
ROOT_DIR = CONFIG_DIR.parents[2] # Navigates up 3 levels to project root

# Paths to YAML configs
MODEL_YAML_PATH = ROOT_DIR / "config" / "model.yaml"
CLASS_YAML_PATH = ROOT_DIR / "config" / "class_mapping.yaml"

# Load model configuration
try:
    with MODEL_YAML_PATH.open("r") as f:
        model_config = yaml.safe_load(f)
        if model_config is None:
            raise ValueError(f"Configuration file {MODEL_YAML_PATH} is empty.")
except FileNotFoundError as e:
    raise FileNotFoundError(f"Model configuration file not found at: {MODEL_YAML_PATH}") from e
except yaml.YAMLError as e:
    raise ValueError(f"Failed to parse YAML file at {MODEL_YAML_PATH}: {e}") from e

# Load class mapping configuration
try:
    with CLASS_YAML_PATH.open("r") as f:
        class_mapping = yaml.safe_load(f)
        if class_mapping is None:
            raise ValueError(f"Class mapping file {CLASS_YAML_PATH} is empty.")

except FileNotFoundError as e:
    raise FileNotFoundError(f"Class mapping file not found at: {CLASS_YAML_PATH}") from e
except yaml.YAMLError as e:
    raise ValueError(f"Failed to parse YAML file at {CLASS_YAML_PATH}: {e}") from e

# Extract and validate values from configuration dictionaries
try:
    IMAGE_SIZE = model_config["model"]["image_size"]
except KeyError as e:
    raise KeyError(f"Missing required key 'image_size' under 'model' in {MODEL_YAML_PATH}") from e

try:
    rel_model_path = model_config["weights"]["path"]
    MODEL_PATH = ROOT_DIR / rel_model_path
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model weights not found at: {MODEL_PATH}"
        )
except KeyError as e:
    raise KeyError(f"Missing required key 'path' under 'weights' in {MODEL_YAML_PATH}") from e

try:
    # Ensure keys are sorted numerically to map correct index to class name
    sorted_keys = sorted(class_mapping.keys(), key=lambda x: int(x))
    CLASS_NAMES = [class_mapping[k] for k in sorted_keys]
except (ValueError, TypeError) as e:
    raise TypeError(f"Keys in {CLASS_YAML_PATH} must be integer indices. Error: {e}") from e

# Generate display names dynamically for presentation in UI
DISPLAY_NAMES = {
    name: name.replace('_', ' ').replace('notumor', 'no tumor').title()
    for name in CLASS_NAMES
}


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

