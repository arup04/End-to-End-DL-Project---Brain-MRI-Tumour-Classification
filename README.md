# 🧠 Brain MRI Tumor Classification with Explainable AI (Grad-CAM++)

An end-to-end Deep Learning project that classifies brain MRI scans into **four distinct tumor categories** using **EfficientNet-B2** and provides visual model interpretability via **Grad-CAM++** heatmaps, all served through a production-grade **Streamlit** web application.

The project is architected for production from the ground up — featuring a fully containerized, ultra-fast **Docker** deployment powered by the `uv` package manager, with a complete manual deployment walkthrough to **AWS ECR + EC2**.

---

## 🩻 What the Model Can Detect

The classifier categorizes a brain MRI scan into one of four mutually exclusive classes:

| Index | Class Key    | Display Name | Color Code |
|-------|--------------|--------------|------------|
| 0     | `glioma`     | Glioma       | 🟦 Blue    |
| 1     | `meningioma` | Meningioma   | 🟨 Amber   |
| 2     | `notumor`    | No Tumor     | 🟩 Green   |
| 3     | `pituitary`  | Pituitary    | 🟥 Red     |

> **⚕️ Medical Disclaimer:** This tool is for **educational and research purposes only**. It does **not** provide medical diagnoses. Always consult a qualified healthcare professional for any clinical decision.

---

## ✨ Key Features

- **EfficientNet-B2 Classifier** — A state-of-the-art convolutional neural network fine-tuned on the 4-class brain MRI dataset.
- **Grad-CAM++ Explainability** — Visual saliency maps highlight exactly which regions of the scan influenced the model's decision.
- **Skull Masking** — Background noise is removed by detecting and isolating the largest connected component (the brain) before rendering the heatmap.
- **Adaptive Overlay Opacity** — Heatmap alpha blending is dynamically scaled based on activation strength above a user-controlled threshold.
- **Interactive Streamlit Dashboard** — Two-page UI with Scan Analysis and Model Performance & Analytics views.
- **Optuna Hyperparameter Optimization** — Best hyperparameters found via automated Bayesian search.
- **Production Docker Build** — Multi-stage-style build with `uv` and layer-caching.
- **AWS Deployment (Manual)** — Successfully deployed to EC2 via ECR with full troubleshooting documented below.

---

## 🖼️ Screenshots

### Page 1 — Scan Analysis (Upload Screen)
![App home — Scan Analysis upload screen](assets/image%201.png)

### Page 2 — Model Performance & Analytics (Live on AWS EC2)
![Model Performance and Analytics page running live on AWS EC2](assets/image%202.png)

### Prediction Result — Meningioma (99.99% Confidence)
![Prediction result showing Meningioma classified with 99.99% confidence](assets/image%203.png)

### Grad-CAM++ Heatmap & Overlay
![Grad-CAM++ heatmap and MRI overlay highlighting tumor region](assets/image%204.png)

### Confidence Banner & Medical Disclaimer
![High confidence banner and medical disclaimer displayed after prediction](assets/image%205.png)

---

## 📊 Dataset

This project uses the **[Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)** by **Masoud Nickparvar**, publicly available on Kaggle.

### Overview

| Property         | Details                                                    |
|------------------|------------------------------------------------------------|
| **Source**       | [Kaggle — Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) |
| **Author**       | Masoud Nickparvar                                          |
| **Total Images** | ~7,023 MRI scans                                           |
| **Classes**      | 4 (Glioma, Meningioma, No Tumor, Pituitary)                |
| **Format**       | JPEG images of varying resolutions                         |
| **Split**        | Pre-divided into `Training/` and `Testing/` folders        |

### Class Distribution (Approximate)

| Class       | Training Images | Testing Images |
|-------------|-----------------|----------------|
| Glioma      | ~1,321          | ~300           |
| Meningioma  | ~1,339          | ~306           |
| No Tumor    | ~1,595          | ~405           |
| Pituitary   | ~1,457          | ~300           |

### Download Instructions

The dataset is **not included** in this repository (excluded via `.gitignore`). Download it and place it under `artifacts/data/`:

```bash
# Install Kaggle CLI (if not already installed)
pip install kaggle

# Download the dataset (requires Kaggle API key in ~/.kaggle/kaggle.json)
kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset -p artifacts/data/ --unzip
```

After extraction, the folder structure should look like:

```
artifacts/data/
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/
```

> **Note:** Training augmentation (random horizontal flips, rotations, color jitter) was applied to improve generalization. See the training notebook for full details.

---

## 📂 Project & Folder Structure

```text
End-to-End-DL-Project---Brain-MRI-Tumour-Classification/
│
├── .github/
│   └── workflows/                       # Placeholder for future GitHub Actions CI/CD
│       └── .gitkeep
│
├── artifacts/
│   ├── logs/                            # Training outputs & evaluation artifacts
│   │   ├── .gitkeep
│   │   ├── project_summary.json         # Test accuracy, hyperparams, classification report
│   │   ├── final_learning_curve.png     # Loss & accuracy curves from final training run
│   │   ├── confusion_matrix_optuna_final.png
│   │   └── Te-meTr_0009.jpg             # Sample MRI image used for testing
│   └── models/
│       ├── .gitkeep
│       ├── best_model.pth               # Trained EfficientNet-B2 weights (git-tracked)
│       └── optuna_study.pkl             # Serialized Optuna study for HPO reproducibility
│
├── config/
│   ├── model.yaml                       # Architecture name, image size, weights path
│   └── class_mapping.yaml              # Integer index → class name mapping (0→glioma…)
│
├── notebooks/
│   ├── .gitkeep
│   └── Brain_MRI_Training.ipynb        # End-to-end training, Optuna HPO, evaluation
│
├── src/
│   └── brain_mri/                      # Installable Python package (src-layout)
│       ├── __init__.py
│       ├── cloud/                      # Cloud integrations (S3 placeholder)
│       │   └── __init__.py
│       ├── config/                     # Path resolution & constants
│       │   ├── __init__.py
│       │   └── configuration.py        # Loads YAMLs, exposes MODEL_PATH, CLASS_NAMES, DEVICE…
│       ├── entity/                     # Config dataclasses
│       │   └── __init__.py
│       ├── inference/                  # Inference & Explainable AI pipeline
│       │   ├── __init__.py
│       │   ├── gradcam.py              # Grad-CAM++ with skull masking & adaptive overlay
│       │   └── predictor.py            # Model loader, image preprocessor, forward pass
│       └── utils/                      # Shared utility helpers
│           └── __init__.py
│
├── tests/
│   ├── __init__.py
│   └── test_predictor.py               # Unit tests: preprocessing, inference, model loading
│
├── web/
│   └── app.py                          # Streamlit dashboard (558 lines, two-page UI)
│
├── assets/                             # Screenshots & demo images for README
│   ├── image 1.png                     # App home — Scan Analysis upload screen
│   ├── image 2.png                     # Model Performance & Analytics page (AWS live)
│   ├── image 3.png                     # Prediction result — Meningioma 99.99%
│   ├── image 4.png                     # Grad-CAM++ heatmap + overlay
│   └── image 5.png                     # Confidence banner & medical disclaimer
│
├── .dockerignore                        # Excludes .venv, raw data, secrets from Docker context
├── .env.example                         # Template for AWS credentials & EC2 config
├── .gitignore                           # Excludes .venv, raw datasets, .env secrets
├── .python-version                      # Pins Python 3.12 for uv/pyenv
├── Dockerfile                           # Production container (python:3.12-slim + uv)
├── LICENSE                              # MIT License
├── pyproject.toml                       # PEP 621 project metadata & locked dependencies (uv)
├── README.md                            # This file
└── uv.lock                              # Fully reproducible dependency graph
```


---

## 🏗️ Architecture Deep Dive

### 1. Neural Network — EfficientNet-B2

The model is sourced from `torchvision.models` and its classification head is replaced:

```python
# Original head: nn.Linear(in_features, 1000)
# Replaced with:
model.classifier[1] = nn.Linear(num_features, 4)   # 4 tumor classes
```

- **Input:** `260 × 260 px` RGB image
- **Preprocessing:** Resize (Lanczos) → ToTensor → Normalize (ImageNet mean/std `[0.485, 0.456, 0.406]` / `[0.229, 0.224, 0.225]`)
- **Output:** Softmax probabilities over 4 classes

---

### 2. Explainable AI — Grad-CAM++

Implemented from scratch in `src/brain_mri/inference/gradcam.py`.

**Why Grad-CAM++ over vanilla Grad-CAM?**
Grad-CAM++ (Chattopadhyay et al., 2018) weights each spatial location's gradient contribution using **second- and third-order derivative terms**. This produces tighter, more compact heatmaps — critical for small, localized regions like tumors.

**Core Algorithm:**
```
α_c_k  = ∑ (∂²y^c / ∂A^k_ij²) / (2·∑ (∂²y^c / ∂A^k_ij²) + ∑ A^k_ij · ∂³y^c / ∂A^k_ij³)
L_GradCAM++ = ReLU(∑_k  w_k^c · A^k)
```

**Post-processing Pipeline:**
1. **Upsample** raw CAM from feature map resolution → 260×260 px (bicubic)
2. **Gaussian Blur** for smooth heatmap edges
3. **Skull Masking** via connected-component analysis to isolate the brain
4. **Gamma Sharpening** to suppress weak activations (`cam = cam^γ`)
5. **Adaptive Overlay** — per-pixel alpha blending controlled by user sliders

---

### 3. Configuration System

All settings flow from two YAML files loaded at import time:

**`config/model.yaml`**
```yaml
model:
  architecture: efficientnet_b2
  num_classes: 4
  image_size: 260

weights:
  path: artifacts/models/best_model.pth
```

**`config/class_mapping.yaml`**
```yaml
0: glioma
1: meningioma
2: notumor
3: pituitary
```

---

### 4. Streamlit Web Application — `web/app.py`

A **two-page dashboard** with 558 lines of fully modularized code.

- **Page 1: 🔍 Scan Analysis** — Upload MRI, get prediction, probability bars, and Grad-CAM++ heatmap.
- **Page 2: 📊 Model Performance** — Displays test accuracy, confusion matrix, learning curves, and hyperparameter tables.
- **Sidebar Controls** — Heatmap sensitivity and sharpness sliders.
- **Model Caching** — `@st.cache_resource` ensures the model loads only once per session.

---

## 🐳 Dockerization & Production Optimization

The `Dockerfile` is engineered around four key optimizations:

### 1. `uv` Package Manager for Layer Caching

```dockerfile
# Dependencies installed BEFORE copying source code
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
# Source copied last — doesn't invalidate the dependency layer
COPY . .
```

**Result:** Rebuilds after code changes take seconds, not minutes.

### 2. Headless OpenCV

`opencv-python-headless` eliminates all X11/Qt GUI dependencies, preventing `Segmentation Fault (exit code 139)` crashes caused by PyArrow/Streamlit triggering Qt initialization in a headless environment.

### 3. OpenMP Support

```dockerfile
RUN apt-get install -y libgomp1
```

`libgomp1` provides thread-parallelism for PyTorch CPU operations and OpenCV kernels.

### 4. Environment Variables

```dockerfile
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    STREAMLIT_SERVER_HEADLESS=true
```

### Running with Docker

```bash
docker build -t brain-mri-app .
docker run -p 8501:8501 brain-mri-app
# Open http://localhost:8501
```

---

## ☁️ AWS Deployment: Step-by-Step Walkthrough

The project was successfully deployed to AWS EC2 using ECR as the container registry. Below is the exact manual deployment process, including troubleshooting.

### Infrastructure Specifications

| Component           | Service Used    | Configuration                     |
|---------------------|-----------------|-----------------------------------|
| Container Registry  | Amazon ECR      | Private repository in `ap-south-1`|
| Compute             | Amazon EC2      | `c7i-flex.large` (2 vCPU, 4 GB RAM) |
| Storage             | EBS             | 30 GB gp3                         |
| OS                  | Ubuntu          | Ubuntu 26.04 LTS                  |
| Networking          | Security Groups | SSH (22), Streamlit (8501)        |

### Step 1: Install & Configure AWS CLI (Local)

```bash
aws --version
# aws-cli/2.36.2

aws configure
# AWS Access Key ID: AKIA...
# AWS Secret Access Key: ...
# Default region: ap-south-1
# Output format: json

aws sts get-caller-identity
# {
#     "UserId": "AIDAXO57KITY3V3IZJJIP",
#     "Account": "<YOUR_AWS_ACCOUNT_ID>",
#     "Arn": "arn:aws:iam::<YOUR_AWS_ACCOUNT_ID>:user/firstproj"
# }
```

### Step 2: Create ECR Repository & Push Docker Image

```bash
# Create repository
aws ecr create-repository --repository-name brain-mri --region ap-south-1
# Output: <YOUR_AWS_ACCOUNT_ID>.dkr.ecr.ap-south-1.amazonaws.com/brain-mri

# Login to ECR
aws ecr get-login-password --region ap-south-1 | docker login --username AWS --password-stdin <YOUR_AWS_ACCOUNT_ID>.dkr.ecr.ap-south-1.amazonaws.com
# Login Succeeded

# Tag and push
docker tag brain-mri-app:latest <YOUR_AWS_ACCOUNT_ID>.dkr.ecr.ap-south-1.amazonaws.com/brain-mri:latest
docker push <YOUR_AWS_ACCOUNT_ID>.dkr.ecr.ap-south-1.amazonaws.com/brain-mri:latest
# latest: digest: sha256:c030e9a5c576... size: 856
```

### Step 3: Launch EC2 Instance

| Setting        | Value                                           |
|----------------|-------------------------------------------------|
| Region         | Mumbai (`ap-south-1`)                           |
| AMI            | Ubuntu 26.04 LTS                                |
| Instance Type  | `c7i-flex.large` (2 vCPU, 4 GB RAM)            |
| Key Pair       | `brain-mri.pem`                                 |
| Storage        | 30 GB gp3                                       |
| Security Group | SSH (22) + Custom TCP (8501) open to `0.0.0.0/0`|

### Step 4: Connect to EC2 (SSH Troubleshooting)

**Initial Attempt:**
```bash
ssh -i "D:\Cloud AWS\brain-mri.pem" ubuntu@<YOUR_EC2_PUBLIC_IP>
# Error: UNPROTECTED PRIVATE KEY FILE
```

**Fix Applied:**
1. Right-click `brain-mri.pem` → **Properties** → **Security** → **Advanced**
2. Disable inheritance
3. Remove inherited permissions
4. Grant **Full Control** only to the current user

**Successful Connection:**
```bash
ssh -i "D:\Cloud AWS\brain-mri.pem" ubuntu@<YOUR_EC2_PUBLIC_IP>
# Welcome to Ubuntu 26.04 LTS...
```

### Step 5: Install Docker & AWS CLI on EC2

```bash
# Update and install Docker
sudo apt update
sudo apt install -y docker.io
sudo systemctl start docker
sudo systemctl enable docker

# Add user to docker group (exit and reconnect after)
sudo usermod -a -G docker ubuntu

# Install AWS CLI
sudo apt install -y awscli
```

### Step 6: Authenticate ECR from EC2 & Pull Image

**Initial Error:**
```bash
aws ecr get-login-password --region ap-south-1 | docker login ...
# Unable to locate credentials. You can configure credentials by running "aws configure".
```

**Fix:** Run `aws configure` on the EC2 instance with the same Access/Secret keys.

**Successful Pull & Run:**
```bash
aws configure
# Enter keys, region: ap-south-1, output: json

aws ecr get-login-password --region ap-south-1 | docker login --username AWS --password-stdin <YOUR_AWS_ACCOUNT_ID>.dkr.ecr.ap-south-1.amazonaws.com
# Login Succeeded

docker pull <YOUR_AWS_ACCOUNT_ID>.dkr.ecr.ap-south-1.amazonaws.com/brain-mri:latest

docker run -d -p 8501:8501 --name brain-mri-app <YOUR_AWS_ACCOUNT_ID>.dkr.ecr.ap-south-1.amazonaws.com/brain-mri:latest
# Container ID: 1ec9bc9343dc...
```

### Step 7: Verify Deployment

```bash
docker ps
# CONTAINER ID   STATUS          PORTS                                         NAMES
# 1ec9bc9343dc   Up 31 seconds   0.0.0.0:8501->8501/tcp, [::]:8501->8501/tcp   brain-mri-app

docker logs brain-mri-app
# 2026-07-20 18:08:44.709 Uvicorn server started on 0.0.0.0:8501
# External URL: http://<YOUR_EC2_PUBLIC_IP>:8501
```

> **Live App URL:** http://<YOUR_EC2_PUBLIC_IP>:8501

### Step 8: Cost Optimization & Cleanup

After successful validation, all resources were terminated to save AWS credits:

```bash
docker stop brain-mri-app
docker rm brain-mri-app
```

- Terminated the EC2 instance via AWS Console.
- Deleted the ECR repository to avoid storage costs.

> **Status:** The instance is currently terminated. It can be redeployed in ~10 minutes by following the steps above.

---

## 🔬 Hyperparameter Optimization with Optuna

Automated Bayesian hyperparameter search was performed using **[Optuna](https://optuna.org/)** to find the optimal training configuration for EfficientNet-B2.

### Optuna Configuration

| Setting             | Value                                      |
|---------------------|--------------------------------------------|
| **Sampler**         | `TPESampler` (Tree-structured Parzen Estimator) |
| **Pruner**          | `MedianPruner` (n_startup_trials=5, n_warmup_steps=5) |
| **Direction**       | Minimize validation loss                   |
| **Total Trials**    | 10                                         |
| **Max Epochs / Trial** | 3 (with early pruning)                 |
| **Seed**            | 42 (for reproducibility)                   |
| **Study Name**      | `efficientnet_b2_hpt`                      |

### Search Space

```python
backbone_lr      = trial.suggest_float('backbone_lr', 1e-6, 1e-4, log=True)
head_lr          = trial.suggest_float('head_lr', 1e-4, 1e-2, log=True)
weight_decay     = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)
batch_size       = trial.suggest_categorical('batch_size', [32, 64])
unfreeze_blocks  = trial.suggest_int('unfreeze_blocks', 2, 4)
scheduler_patience = trial.suggest_int('scheduler_patience', 2, 5)
```

### Optimizer Setup (Dual Learning Rate)

Each trial used **AdamW** with separate learning rates for backbone and classification head — a standard fine-tuning best practice:

```python
optimizer = torch.optim.AdamW([
    {'params': backbone_params, 'lr': backbone_lr},   # Slow — pretrained features
    {'params': head_params,     'lr': head_lr},        # Fast — new classification head
], weight_decay=weight_decay)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.1, patience=scheduler_patience
)
```

### Experiment Tracking

All 10 trial runs were tracked individually in **Weights & Biases (W&B)** and grouped under a single sweep. Per-trial metrics logged: `train_loss`, `train_acc`, `val_loss`, `val_acc` per epoch.

Optuna visualizations logged to W&B:
- Optimization history
- Parameter importances
- Parallel coordinate plot
- Slice plots & contour plots
- EDF (Empirical Distribution Function) plot

The full study was serialized and saved to `artifacts/models/optuna_study.pkl` via `joblib` for reproducibility.

---

## 🧠 Model Performance & Results

After HPO, the best trial's hyperparameters were used to retrain the final model on a **90/10 train/val split** for **15 epochs**.

### Best Hyperparameters (from Optuna)

| Hyperparameter         | Search Range             | Best Value    |
|------------------------|--------------------------|---------------|
| Backbone Learning Rate | `1e-6` → `1e-4` (log)   | `5.40e-05`    |
| Head Learning Rate     | `1e-4` → `1e-2` (log)   | `1.59e-03`    |
| Weight Decay           | `1e-5` → `1e-3` (log)   | `2.61e-04`    |
| Batch Size             | `{32, 64}`               | `64`          |
| Unfrozen Blocks        | `2` → `4`                | `4`           |
| Scheduler Patience     | `2` → `5`                | `2`           |

### Final Training Summary

| Metric                  | Value                |
|-------------------------|----------------------|
| Architecture            | EfficientNet-B2      |
| Image Size              | 260 × 260 px         |
| Final Training Epochs   | 15                   |
| Best Validation Loss    | **0.0725**           |
| **Test Accuracy**       | **99.08%**           |
| Explainability          | Grad-CAM++           |

### Per-Class Classification Report (Test Set — 1,311 images)

| Class       | Precision | Recall  | F1-Score | Support |
|-------------|-----------|---------|----------|---------|
| Glioma      | 99.66%    | 97.67%  | 98.65%   | 300     |
| Meningioma  | 97.42%    | 98.69%  | 98.05%   | 306     |
| No Tumor    | 99.75%    | 100.00% | 99.88%   | 405     |
| Pituitary   | 99.34%    | 99.67%  | 99.50%   | 300     |
| **Macro Avg** | **99.04%** | **99.01%** | **99.02%** | **1311** |



## 🔭 Future Roadmap: CI/CD Automation

A GitHub Actions workflow is planned (and partially defined in `.github/workflows/`) to automate the entire pipeline:

1. **Trigger** — Push to `main` branch
2. **Build** — Docker image built in the CI runner
3. **Push** — Image pushed to AWS ECR
4. **Deploy** — SSH into EC2, pull latest image, restart container

**Required GitHub Secrets:**
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `AWS_REGION`
- `AWS_ACCOUNT_ID`
- `ECR_REPOSITORY`
- `ECR_REPOSITORY_URI`
- `EC2_PUBLIC_IP`

---

## 🚀 Getting Started (Local Development)

### Prerequisites

- **Python 3.12+** (pinned in `.python-version`)
- **[`uv`](https://github.com/astral-sh/uv)** — Fast Python package manager

### Installation & Running

```bash
# 1. Clone the repository
git clone https://github.com/arup04/End-to-End-DL-Project---Brain-MRI-Tumour-Classification.git
cd End-to-End-DL-Project---Brain-MRI-Tumour-Classification

# 2. Install dependencies
uv sync

# 3. Place the trained model weights in artifacts/models/best_model.pth

# 4. Run the Streamlit dashboard
uv run streamlit run web/app.py

# 5. Run tests
uv run pytest
```

---

## 📦 Dependencies

The project uses a **fully pinned `uv.lock`** for reproducible builds.

| Package                    | Version   | Role                              |
|----------------------------|-----------|-----------------------------------|
| `torch`                    | 2.13.0    | Deep learning inference           |
| `torchvision`              | 0.28.0    | EfficientNet model zoo            |
| `streamlit`                | 1.59.1    | Web dashboard                     |
| `opencv-python-headless`   | 5.0.0.93  | Heatmap rendering & skull masking |
| `Pillow`                   | 12.3.0    | Image I/O                         |
| `numpy`                    | 2.5.1     | Array operations                  |
| `pandas`                   | 3.0.3     | Analytics tables in dashboard     |
| `pyyaml`                   | 6.0.3     | YAML config loading               |
| `pytest`                   | 9.1.1     | Unit testing                      |

> **PyTorch Source:** Installed from the CPU-only index (`https://download.pytorch.org/whl/cpu`) for a lighter production image.

---

## 🛠️ Technologies Used

| Category         | Tools & Libraries                       |
|------------------|-----------------------------------------|
| Deep Learning    | PyTorch, Torchvision, EfficientNet-B2   |
| Explainable AI   | Grad-CAM++                              |
| Web Framework    | Streamlit                               |
| Image Processing | OpenCV, Pillow                          |
| Package Manager  | `uv`                                    |
| Containerization | Docker                                  |
| Cloud Services   | AWS EC2, AWS ECR                        |
| Configuration    | YAML                                    |
| Testing          | Pytest                                  |
| Language         | Python 3.12+                            |

---

## 📌 Quick Reference: AWS Commands Cheatsheet

| Operation        | Command                                                                                         |
|------------------|-------------------------------------------------------------------------------------------------|
| ECR Login (Local)| `aws ecr get-login-password --region ap-south-1 \| docker login --username AWS --password-stdin <YOUR_AWS_ACCOUNT_ID>.dkr.ecr.ap-south-1.amazonaws.com` |
| Tag Image        | `docker tag brain-mri-app:latest <YOUR_AWS_ACCOUNT_ID>.dkr.ecr.ap-south-1.amazonaws.com/brain-mri:latest` |
| Push Image       | `docker push <YOUR_AWS_ACCOUNT_ID>.dkr.ecr.ap-south-1.amazonaws.com/brain-mri:latest`                   |
| SSH to EC2       | `ssh -i "brain-mri.pem" ubuntu@<PUBLIC_IP>`                                                     |
| ECR Login (EC2)  | `aws ecr get-login-password --region ap-south-1 \| docker login --username AWS --password-stdin <YOUR_AWS_ACCOUNT_ID>.dkr.ecr.ap-south-1.amazonaws.com` |
| Pull Image       | `docker pull <YOUR_AWS_ACCOUNT_ID>.dkr.ecr.ap-south-1.amazonaws.com/brain-mri:latest`                   |
| Run Container    | `docker run -d -p 8501:8501 --name brain-mri-app <YOUR_AWS_ACCOUNT_ID>.dkr.ecr.ap-south-1.amazonaws.com/brain-mri:latest` |
| Check Logs       | `docker logs brain-mri-app`                                                                     |

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📄 License

This project is licensed under the **MIT License**.

---

## 🙏 Acknowledgments

- **EfficientNet-B2** for feature extraction
- **Grad-CAM++** for explainability
- **Streamlit** for rapid web dashboard development
- **AWS** for cloud deployment infrastructure
- The open-source community

---

## 👤 Author

**Arup Das**  
Built with PyTorch · EfficientNet-B2 · Grad-CAM++ · Streamlit · Docker · AWS

---

⭐ If you found this project helpful, please give it a star on GitHub!
