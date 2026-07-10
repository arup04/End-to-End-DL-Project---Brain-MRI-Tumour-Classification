import os
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='[%(asctime)s]: %(message)s'
)

logger = logging.getLogger(__name__)

# List of all files and folders to create
list_of_files = [
    # GitHub Actions
    ".github/workflows/.gitkeep",
    
    # Artifacts
    "artifacts/data/.gitkeep",
    "artifacts/models/.gitkeep",
    "artifacts/logs/.gitkeep",
    
    # Config
    "config/config.yaml",
    
    # Notebooks
    "notebooks/.gitkeep",
    
    # Source Code Package
    "src/brain_mri/__init__.py",
    "src/brain_mri/config/__init__.py",
    "src/brain_mri/config/configuration.py",
    "src/brain_mri/entity/__init__.py",
    "src/brain_mri/entity/config_entity.py",
    "src/brain_mri/inference/__init__.py",
    "src/brain_mri/inference/predictor.py",
    "src/brain_mri/inference/gradcam.py",
    "src/brain_mri/cloud/__init__.py",
    "src/brain_mri/cloud/aws_s3.py",
    "src/brain_mri/utils/__init__.py",
    "src/brain_mri/utils/common.py",
    
    # Tests
    "tests/__init__.py",
    "tests/test_predictor.py",
    
    # Web App
    "web/app.py",
    
    # Root Files
    ".gitignore",
    ".env.example",
    "Dockerfile",
    "requirements.txt",
    "setup.py",
    "README.md",
]

# Create all files and folders
for filepath in list_of_files:
    filepath = Path(filepath)
    
    # Get directory and filename
    filedir, filename = os.path.split(filepath)
    
    # Create directory if it doesn't exist
    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logger.info(f"Created directory: {filedir}")
    
    # Create file if it doesn't exist or is empty
    if not os.path.exists(filepath):
        with open(filepath, "w") as f:
            pass
        logger.info(f"Created file: {filepath}")
    elif os.path.getsize(filepath) == 0:
        logger.info(f"File already exists (empty): {filepath}")
    else:
        logger.info(f"File already exists (not empty): {filepath}")

logger.info("\n✅ All files and folders created successfully!")