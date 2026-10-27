# -*- coding: utf-8 -*-
"""
DermAI — Cloud Model Training Pipeline v1.0
Optimized for Google Colab T4 GPU Execution Rings
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
import kagglehub

# 1. SET THE COMPUTATION ENGINE TO THE CLOUD GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using processing device: {device}")

# 2. DATA AUGMENTATION PIPELINE (Optimized for EfficientNet-B4)
transform = transforms.Compose([
    transforms.Resize((380, 380)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 3. DIRECT CLOUD CACHE DOWNLOAD PATH FETCHING
print("Locating dataset directory cache...")
path = kagglehub.dataset_download("shubhamgoel27/dermnet")
dataset_path = os.path.join(path, "train")

# Initialize data loaders over the 23-class directory layout
train_dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)

print(f"\nSuccessfully linked dataset!")
print(f"Detected classes ({len(train_dataset.classes)} total): {train_dataset.classes}")
print(f"Total training images: {len(train_dataset)}")

# 4. INITIALIZE PRE-TRAINED EFFICIENTNET-B4 & SWAP OUTPUT LAYER
model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.DEFAULT)
num_ftrs = model.classifier[1].in_features

# Dynamically map the projection layer outputs to the total dataset class length
model.classifier[1] = nn.Linear(num_ftrs, len(train_dataset.classes))
model = model.to(device)

# 5. DEFINE OPTIMIZER AND LOSS COEFFICIENTS
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.01)

# 6. START THE TRAINING PIPELINE
print("\n--- STARTING TRAINING PIPELINE ON CLOUD T4 GPU ---")
model.train()
epochs = 3  

for epoch in range(epochs):
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        # Backward propagation pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Print update frames every 20 batches
        if (batch_idx + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] | Batch [{batch_idx+1}/{len(train_loader)}] | Loss: {loss.item():.4f}")

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100.0 * correct / total
    print(f"--> EPOCH {epoch+1} COMPLETE | Avg Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.2f}%\n")

# 7. EXPORT MODEL WEIGHT FILE
print("Training complete! Moving model to CPU memory layout...")
torch.save(model.to('cpu').state_dict(), 'model_weights.pth')
print("SUCCESS: 'model_weights.pth' successfully generated and written to disk.")