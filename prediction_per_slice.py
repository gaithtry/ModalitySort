import os
import sys
import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    ScaleIntensityRanged,
    Resized,
)

class_names = ['CTA', 'FLAIR', 'MRA', 'T1w', 'T2w', 'ncCT'] #make sure same order as training class order

# Model
model = models.densenet121(weights=None)  # Set as None since we're loading weights
model.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
num_ftrs = model.classifier.in_features
model.classifier = nn.Linear(num_ftrs, len(class_names))

# Load the trained weights
model_path = "best_metric_model.pth"
if not os.path.exists(model_path):
    print(f"Model weights not found at {model_path}. Please run training.py first.")
    sys.exit(1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

# Transforms for prediction 
predict_transforms = Compose([
    EnsureChannelFirstd(keys="image", channel_dim="no_channel"),
    ScaleIntensityRanged(keys="image", a_min=0, a_max=255,b_min=0.0, b_max=1.0, clip=True),
    Resized(keys="image", spatial_size=(256, 256), mode="bilinear"),
])

def predict_nifti_modality(nifti_path, model, class_names, transforms):
    # Load the NIfTI image
    img = nib.load(nifti_path)
    image_data = img.get_fdata().astype(np.float32)

    # build an inference slice index set that approximates training-level coverage
    depth = image_data.shape[-1]

    # Choose up to 96 slices (or fewer if the volume is small), evenly spaced
    max_slices = 96
    if depth <= max_slices:
        slice_indices = list(range(depth))
    else:
        slice_indices = list(np.linspace(0, depth - 1, max_slices, dtype=int))

    all_probs = []

    with torch.no_grad():
        for idx in slice_indices:
            slice_data = image_data[:, :, idx]

            # Skip slices that are mostly background or empty (if this occurs)
            if np.mean(slice_data) < 1e-3 and np.count_nonzero(slice_data) < 0.01 * slice_data.size:
                continue

            data_dict = {"image": slice_data}
            data_dict = transforms(data_dict)
            slice_tensor = data_dict["image"].unsqueeze(0).to(device)

            output = model(slice_tensor)
            probabilities = F.softmax(output, dim=1)
            all_probs.append(probabilities)

    # slice by slice predictions across all valid slices
    per_slice_results = []
    for slice_idx, probs in zip(slice_indices, all_probs):
        confidence, predicted_idx = torch.max(probs, 1)
        per_slice_results.append({
            'slice': slice_idx,
            'predicted_class': class_names[predicted_idx.item()],
            'confidence': confidence.item()
    })

    # summary of predictions
    mean_probs = torch.mean(torch.stack(all_probs), dim=0)
    confidence, predicted_idx = torch.max(mean_probs, 1)
    summary = {
        'predicted_class': class_names[predicted_idx.item()],
        'confidence': confidence.item()
    }
    return per_slice_results, summary

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py <path_to_nifti_file>")
        sys.exit(1)

    nifti_path = sys.argv[1]
    if not os.path.exists(nifti_path):
        print(f"File not found: {nifti_path}")
        sys.exit(1)

    per_slice_results, summary = predict_nifti_modality(nifti_path, model, class_names, predict_transforms)

    print(f"\nPer-slice predictions:")
    print(f"{'Slice':<10} {'Predicted Class':<20} {'Confidence':<10}")
    print("-" * 40)
    for result in per_slice_results:
        print(f"{result['slice']:<10} {result['predicted_class']:<20} {result['confidence']:.4f}")

    print(f"\nSummary (averaged across {len(per_slice_results)} slices):")
    print(f"Predicted modality: {summary['predicted_class']}")
    print(f"Confidence score:   {summary['confidence']:.4f}")
