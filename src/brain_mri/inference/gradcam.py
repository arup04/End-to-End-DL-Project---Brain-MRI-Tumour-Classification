import os
import sys
import cv2
import numpy as np
import torch
from PIL import Image

# Ensure the package is importable
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from brain_mri.config.configuration import IMAGE_SIZE

class GradCAM:
    """
    Generate Grad-CAM++ heatmaps for model interpretability.
    
    Grad-CAM++ (Chattopadhyay et al., 2018) refines vanilla Grad-CAM by
    weighting each spatial location's gradient contribution using
    second- and third-order derivative terms instead of a plain spatial
    average. In practice this produces tighter, more compact heatmaps
    when the class-relevant region is small and localized (e.g. a
    tumor), instead of spreading activation across a broad area.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hooks_registered = False
        self._register_hooks()

    def _register_hooks(self):
        if not self.hooks_registered:
            self.forward_handle = self.target_layer.register_forward_hook(self.save_activation)
            self.backward_handle = self.target_layer.register_full_backward_hook(self.save_gradient)
            self.hooks_registered = True

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_cam(self, input_tensor, target_class, smooth=True):
        self.model.eval()
        output = self.model(input_tensor)
        self.model.zero_grad()
        class_loss = output[0, target_class]
        class_loss.backward()

        grads = self.gradients[0]        # (C, H, W)
        activations = self.activations[0]  # (C, H, W)

        # ---- Grad-CAM++ weighting ----
        eps = 1e-8
        grads_2 = grads.pow(2)
        grads_3 = grads.pow(3)
        sum_activations = activations.sum(dim=(1, 2), keepdim=True)  # (C, 1, 1)

        alpha_denom = 2.0 * grads_2 + sum_activations * grads_3
        alpha_denom = torch.where(alpha_denom != 0, alpha_denom, torch.full_like(alpha_denom, eps))
        alphas = grads_2 / (alpha_denom + eps)

        # Only positive gradients contribute
        weights = (alphas * torch.relu(grads)).sum(dim=(1, 2), keepdim=True)  # (C, 1, 1)

        cam = (weights * activations).sum(dim=0)
        cam = torch.relu(cam)
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()

        cam = cam.cpu().numpy().astype(np.float32)

        # Upsample with high-quality interpolation
        cam = cv2.resize(cam, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)
        cam = np.clip(cam, 0, 1)

        if smooth:
            cam = cv2.GaussianBlur(cam, (0, 0), sigmaX=IMAGE_SIZE * 0.015)
            cam -= cam.min()
            if cam.max() > 0:
                cam /= cam.max()

        return cam


def get_target_layer(model):
    """
    Select the target layer for Grad-CAM: the LAST layer of model.features.
    """
    return model.features[-1]


def compute_foreground_mask(original_image, size=IMAGE_SIZE):
    """
    Build a binary mask of the actual brain tissue in the scan, so we can
    zero out any Grad-CAM activation that falls on background (black)
    pixels outside the skull.
    """
    gray = np.array(original_image.convert("L").resize((size, size), Image.LANCZOS))

    # Brain tissue is well above the near-black background in these scans
    _, mask = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)

    # Clean up small speckles and fill small holes inside the brain
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Keep only the largest connected component (the head)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        clean_mask = np.zeros_like(mask)
        cv2.drawContours(clean_mask, [largest], -1, 255, thickness=cv2.FILLED)
        mask = clean_mask

    return (mask.astype(np.float32) / 255.0)


def generate_heatmap(gradcam, tensor, pred_class, original_image, gamma=1.8):
    """
    Produce the final display heatmap with skull masking and gamma sharpening.
    """
    tensor_grad = tensor.clone().detach().requires_grad_(True)
    cam = gradcam.generate_cam(tensor_grad, pred_class)

    mask = compute_foreground_mask(original_image)
    cam = cam * mask

    if cam.max() > 0:
        cam = cam / cam.max()

    cam = np.power(cam, gamma)
    return cam


def create_overlay(original_image, heatmap, threshold=0.25, max_alpha=0.75):
    """
    Blend the heatmap onto the MRI with PER-PIXEL opacity that scales with
    activation strength.
    """
    img_np = np.array(original_image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)).astype(np.float32)

    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB).astype(np.float32)

    # Rescale heatmap intensity above threshold to [0, 1], zero below it
    alpha = np.clip((heatmap - threshold) / (1.0 - threshold + 1e-8), 0, 1)
    alpha = alpha * max_alpha
    alpha_3ch = np.repeat(alpha[:, :, np.newaxis], 3, axis=2)

    overlay = img_np * (1 - alpha_3ch) + heatmap_colored * alpha_3ch
    return np.uint8(np.clip(overlay, 0, 255))
