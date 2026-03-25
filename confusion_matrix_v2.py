import os
import glob
import random
from monai import data
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
from monai.transforms import Compose, EnsureChannelFirstd, ScaleIntensityRanged, Resized
from torchvision import models

# Data directory
data_dir = 'nifti_training'

# Define modalities
modalities = sorted(x for x in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, x)))
class_names = modalities

print("Modalities:", modalities)

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Transforms
transforms = Compose([
    EnsureChannelFirstd(keys=['image'], channel_dim='no_channel'),
    ScaleIntensityRanged(keys=['image'], a_min=0, a_max=255, b_min=0, b_max=1, clip=True),
    Resized(keys=['image'], spatial_size=(256, 256), mode='bilinear')
])

class MyDataset(Dataset):
    def __init__(self, files, transform=None):
        self.files = files
        self.transform = transform
        self._cache = {}

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        item = self.files[idx]
        img_path = item['image_path']
        slice_idx = item['slice_idx']
        label = item['label']

        if img_path not in self._cache:
            self._cache[img_path] = nib.load(img_path).get_fdata().astype(np.float32)
        data = self._cache[img_path]


        # Extract slice as numpy array
        slice_np = data[:, :, slice_idx].astype(np.float32)

        if self.transform:
            data_dict = {'image': slice_np}
            transformed = self.transform(data_dict)

            return transformed['image'].float(), label

        # No transform: manually convert numpy array to tensor
        return torch.tensor(slice_np, dtype=torch.float32).unsqueeze(0), label
        

# Load files
files = []
for modality in modalities:
    nii_files = glob.glob(os.path.join(data_dir, modality, '*.nii')) + \
            glob.glob(os.path.join(data_dir, modality, '*.nii.gz'))
    for nii_file in nii_files:
        img = nib.load(nii_file)
        data = img.get_fdata()
        num_slices = data.shape[2]
        for slice_idx in range(num_slices):
            files.append({
                'image_path': nii_file,
                'slice_idx': slice_idx,
                'label': modalities.index(modality)
            })

# Split into train and val (same as training.py)
labels = [f['label'] for f in files]
train_files, val_files = train_test_split(files, test_size=0.2, random_state=42, stratify=labels)

# Create validation dataset and dataloader
val_dataset = MyDataset(val_files, transform=transforms)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

# Load model
model = models.densenet121(weights=None)
model.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
num_ftrs = model.classifier.in_features
model.classifier = nn.Linear(num_ftrs, len(modalities))
model.load_state_dict(torch.load('best_metric_model.pth', map_location=device))
model.to(device)
model.eval()

# Collect predictions
y_true = []
y_pred = []

with torch.no_grad():
    for inputs, labels in val_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

# Convert to numpy
y_true = np.array(y_true)
y_pred = np.array(y_pred)

# Randomly select 9 indices from the validation set
sample_indices = random.sample(range(len(val_files)), min(9, len(val_files)))

plt.figure(figsize=(10, 10))
for i, idx in enumerate(sample_indices):
    # Retrieve the corresponding item from val_files
    item = val_files[idx]

    # Load the full 3D NIfTI image
    img_3d = nib.load(item['image_path'])
    image_data_3d = img_3d.get_fdata()

    # Extract the 2D slice
    slice_2d = image_data_3d[:, :, item['slice_idx']]

    # Get true and predicted labels
    true_label = y_true[idx]
    predicted_label = y_pred[idx]

    # Map numerical labels to class names
    true_class_name = class_names[true_label]
    predicted_class_name = class_names[predicted_label]

    # Display the 2D slice
    plt.subplot(3, 3, i + 1)
    plt.imshow(slice_2d, cmap='gray', vmin=image_data_3d.min(), vmax=image_data_3d.max())
    plt.title(f"Ground Truth: {true_class_name}\nPred: {predicted_class_name}")
    plt.xticks([])
    plt.yticks([])

# Adjust subplot parameters for a tight layout
plt.tight_layout()

# Save the plot instead of showing
plt.savefig('sample_predictions.png')
print("Sample predictions saved to sample_predictions.png")

# Calculate the confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

# Display the confusion matrix
fig, ax = plt.subplots(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap=plt.cm.Blues, ax=ax)
ax.set_xticks(range(len(class_names)))
ax.set_xticklabels(class_names, rotation=45, ha='right')
ax.set_yticks(range(len(class_names)))
ax.set_yticklabels(class_names)
plt.title("Confusion Matrix for 2D Slice Classification")
plt.savefig('confusion_matrix.png')
print("Confusion matrix saved to confusion_matrix.png")

# Print classification report for more detailed metrics
print("\nClassification Report:")
print(classification_report(y_true, y_pred, labels=list(range(len(class_names))), target_names=class_names))
