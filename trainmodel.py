import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# --- Configuration ---
IMG_PATH = "images/frame_0000.jpg"
MASK_PATH = "masks_binary/frame_0000.jpg"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Dataset Class ---
class SingleSampleDataset(Dataset):
    def __init__(self, img_path, mask_path, transform=None):
        self.img_path = img_path
        self.mask_path = mask_path
        self.transform = transform

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        img = Image.open(self.img_path).convert("L")
        mask = Image.open(self.mask_path).convert("L")
        
        if self.transform:
            img = self.transform(img)
            mask = self.transform(mask)
        return img, mask

# --- Dummy UNet Model (replace with yours) ---
class UNetEdgeDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 1, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# --- Transform ---
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# --- Load Data ---
dataset = SingleSampleDataset(IMG_PATH, MASK_PATH, transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# --- Initialize Model ---
model = UNetEdgeDetector().to(DEVICE)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# --- Training Loop ---
model.train()
for epoch in range(5):
    for images, masks in dataloader:
        images, masks = images.to(DEVICE), masks.to(DEVICE)
        preds = model(images)

        loss_map = (preds - masks) ** 2
        loss = criterion(preds, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1} | Loss: {loss.item():.6f}")

        # Visual Debug
        pred_np = preds[0][0].detach().cpu().numpy()
        mask_np = masks[0][0].cpu().numpy()
        diff_np = np.abs(pred_np - mask_np)

        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        axs[0].imshow(pred_np, cmap='gray'); axs[0].set_title("Prediction")
        axs[1].imshow(mask_np, cmap='gray'); axs[1].set_title("Mask")
        axs[2].imshow(diff_np, cmap='hot'); axs[2].set_title("Difference")
        plt.show()
