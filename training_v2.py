import os
import glob
import nibabel as nib
import numpy as np
from sklearn.model_selection import train_test_split
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    ScaleIntensityRanged,
    Resized,
    RandFlipd,
    RandRotate90d,
    RandAffined,
    RandGaussianNoised,
    RandAdjustContrastd,
    RandZoomd,
)
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from tqdm import tqdm

# Unzip if needed (assuming nifti_training.zip exists)
import zipfile
if os.path.exists('nifti_training.zip'):
    with zipfile.ZipFile('nifti_training.zip', 'r') as zip_ref:
        zip_ref.extractall('.')
    print("Extracted nifti_training.zip")

##Loading Nifti Data (load from folder)

root_dir = "nifti_training"
ls = os.listdir(root_dir)

# Get a list of all .nii files in the nifti_training directory and its subdirectories
nifti_files = glob.glob(os.path.join(root_dir, "**", "*.nii"), recursive=True) + glob.glob(os.path.join(root_dir, "**", "*.nii.gz"), recursive=True)
print(f"Found {len(nifti_files)} NIfTI files to process.")

# Create a list to store loaded image data
loaded_images_data = []

# Iterate over the found NIfTI files
for file_path in nifti_files:
  try:
    # Load the NIfTI image
    im = nib.load(file_path)
    image_data = im.get_fdata()
    loaded_images_data.append(image_data)

    print(f"Successfully loaded '{file_path}' with shape {image_data.shape}")

  except Exception as e:
    print(f"Error processing file '{file_path}': {e}")

# Data loading
root_dir = "nifti_training"
class_names = sorted(x for x in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, x)))
num_class = len(class_names)
image_files = [
    [os.path.join(root_dir, class_names[i], x) for x in os.listdir(os.path.join(root_dir, class_names[i])) if not x.startswith('.DS_Store')]
    for i in range(num_class)
]
num_each = [len(image_files[i]) for i in range(num_class)]
image_files_list = []
image_class = []
for i in range(num_class):
    image_files_list.extend(image_files[i])
    image_class.extend([i] * num_each[i])
num_total = len(image_class)

print(f"Total image count: {num_total}")
print(f"Label names: {class_names}")
print(f"Label counts: {num_each}")

# Create data_dicts
data_dicts = [{
    "image": img_path,
    "label": img_class
} for img_path, img_class in zip(image_files_list, image_class)]

# Create slice_data_dicts
slice_data_dicts = []
for entry in data_dicts:
    image_path = entry['image']
    label = entry['label']
    img_3d = nib.load(image_path)
    image_data_3d = img_3d.get_fdata()
    depth = image_data_3d.shape[-1]
    for slice_idx in range(depth):
        slice_dict = {
            'image_path': image_path,
            'slice_idx': slice_idx,
            'label': label
        }
        slice_data_dicts.append(slice_dict)

print(f"Total 2D slices created: {len(slice_data_dicts)}")

# Split
slices_labels = [d['label'] for d in slice_data_dicts]
train_files, val_files, _, _ = train_test_split(
    slice_data_dicts, slices_labels, test_size=0.2, random_state=42, stratify=slices_labels
)

print(f"Number of training slices: {len(train_files)}")
print(f"Number of validation slices: {len(val_files)}")

# Dataset class
class MyDataset(Dataset):
    def __init__(self, data, transforms=None):
        self.data = data
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = item["image_path"]
        slice_idx = item["slice_idx"]
        label = item["label"]
        img_3d = nib.load(image_path)
        image_data_3d = img_3d.get_fdata().astype(np.float32)
        image_data_2d = image_data_3d[:, :, slice_idx]
        data_dict = {"image": image_data_2d, "label": label}
        if self.transforms:
            data_dict = self.transforms(data_dict)
        return data_dict["image"], torch.tensor(data_dict["label"], dtype=torch.long)

# Transforms
train_transforms = Compose([
    EnsureChannelFirstd(keys="image", channel_dim="no_channel"),
    ScaleIntensityRanged(keys="image", a_min=0, a_max=255, b_min=0.0, b_max=1.0, clip=True),
    Resized(keys="image", spatial_size=(256, 256), mode="bilinear"),

    # Geometric augmentations
    RandFlipd(keys="image", prob=0.5, spatial_axis=0),
    RandFlipd(keys="image", prob=0.5, spatial_axis=1),
    RandRotate90d(keys="image", prob=0.5, max_k=3),
    RandAffined(
        keys="image",
        prob=0.4,
        rotate_range=(0.1,),
        translate_range=(10, 10),
        shear_range=(0.05,),
        mode="bilinear",
        padding_mode="zeros",
    ),
    RandZoomd(
        keys="image",
        prob=0.3,
        min_zoom=0.9,
        max_zoom=1.1,
        mode="bilinear",
    ),

    # Intensity augmentations
    RandGaussianNoised(keys="image", prob=0.3, mean=0.0, std=0.05),
    RandAdjustContrastd(keys="image", prob=0.3, gamma=(0.75, 1.25)),
])

val_transforms = Compose([
    EnsureChannelFirstd(keys="image", channel_dim="no_channel"),
    ScaleIntensityRanged(keys="image", a_min=0, a_max=255, b_min=0.0, b_max=1.0, clip=True),
    Resized(keys="image", spatial_size=(256, 256), mode="bilinear"),
])

# Determine best device and DataLoader settings
if torch.cuda.is_available():
    device = torch.device("cuda")
    backend = "cuda"
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    backend = "mps"
else:
    device = torch.device("cpu")
    backend = "cpu"

print("Using device:", device)

use_gpu = torch.cuda.is_available() or torch.backends.mps.is_available()

batch_size = 16 if use_gpu else 8   
num_workers = 2 if backend == "mps" else (4 if backend == "cuda" else 0)
pin_memory = True if backend == "cuda" else False

# Dataloaders
train_ds = MyDataset(data=train_files, transforms=train_transforms)
train_loader = DataLoader(
    train_ds,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=pin_memory,
)

val_ds = MyDataset(data=val_files, transforms=val_transforms)
val_loader = DataLoader(
    val_ds,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=pin_memory,
)

# Model
if __name__ == '__main__':
    print("Starting model initialization...")
    model = models.densenet121(pretrained=False)
    # Modify for 1 channel input and number of classes
    model.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, num_class)

    # Move model to GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"Model moved to device: {device}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=3)
    print("Model initialized successfully!")

    # Training
    num_epochs = 20
    early_stop_patience = 5
    best_accuracy = 0.0
    epochs_no_improve = 0
    best_model_path = "best_metric_model.pth"
    print("Starting training...")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        print(f"\nEpoch {epoch+1}/{num_epochs} starting...")
        for images, labels in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        train_accuracy = 100 * correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Train Accuracy: {train_accuracy:.2f}%")

        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Validation Epoch {epoch+1}"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        val_accuracy = 100 * val_correct / val_total
        print(f"Validation Loss: {val_loss/len(val_loader):.4f}, Validation Accuracy: {val_accuracy:.2f}%")


        scheduler.step(val_accuracy)

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved — val accuracy: {best_accuracy:.2f}%")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve}/{early_stop_patience} epochs")
            if epochs_no_improve >= early_stop_patience:
                print(f"Early stopping at epoch {epoch+1}. Best val accuracy: {best_accuracy:.2f}%")
                break
    

    print("Training completed. Best model saved to", best_model_path)
