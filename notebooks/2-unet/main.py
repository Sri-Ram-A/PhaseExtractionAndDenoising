# %%
"""
## Residual UNET
"""

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
from tqdm import tqdm
import json
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
plt.style.use('dark_background')
print("Using device : ",device)

# %%
# Paths
NOTEBOOK_DIR = Path().resolve()
BASE_DIR = NOTEBOOK_DIR.parents[1]
DATASET_DIR = BASE_DIR / "data"
MODEL_OUTPUT_DIR = BASE_DIR / "models" / "1-residual-block" 
MODEL_OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

INPUT_DIR  = DATASET_DIR / "inputs"
TARGET_DIR = DATASET_DIR / "targets"
HEIGHT , WIDTH = 256 , 256
IMAGE_SIZE = (HEIGHT , WIDTH)
import sys
sys.path.append(str(BASE_DIR))
import helper

for name, directory in [("Input", INPUT_DIR), ("Target", TARGET_DIR)]:
    if directory.is_dir():
        files = list(directory.glob('*'))
        print(f"✅ {name} Directory: {'Contains ' + str(len(files)) + ' files.' if files else 'Empty.'}")
    else:
        print(f"❌ {name} Directory NOT found at: {directory}")

# %%
class InterferogramDataset(Dataset):
    """Simple segmentation dataset for image-mask pairs"""
    
    def __init__(self, image_dir, mask_dir, transform=None, img_size=(512,512)):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform
        self.img_size = img_size
        
        # Get all image files
        valid_exts = {".jpg", ".jpeg", ".png"}
        self.images = sorted([f for f in self.image_dir.iterdir() if f.suffix.lower() in valid_exts])
        self.masks = sorted([f for f in self.mask_dir.iterdir() if f.suffix.lower() in valid_exts])
        
        assert len(self.images) == len(self.masks), "Images and masks count mismatch!"
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Load image and mask
        img = Image.open(self.images[idx]).convert('L')
        mask = Image.open(self.masks[idx]).convert('L')  # Grayscale
        
        # Resize
        img = img.resize(self.img_size, Image.NEAREST)
        mask = mask.resize(self.img_size, Image.NEAREST)
        
        # Convert to numpy
        img = np.array(img).astype(np.float32) / 255.0
        mask = np.array(mask).astype(np.float32) / 255.0
        
        # To tensor (C, H, W)
        img = torch.from_numpy(img).unsqueeze(0)
        mask = torch.from_numpy(mask).unsqueeze(0)  # Add channel dim
        
        return img, mask

# %%
# Create datasets
train_dataset = InterferogramDataset(INPUT_DIR, TARGET_DIR, img_size=IMAGE_SIZE)
BATCH_SIZE = 16
# Split into train/val (80/20)
train_size = int(0.65 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")


# %%
class ResidualBlock(nn.Module):
    """Residual block with two convolutions"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        
        # Skip connection
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        
    def forward(self, x):
        identity = self.skip(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)


# %%
class UNetResidualLite(nn.Module):
    """Lightweight U-Net with residual blocks for denoising / upsampling"""
    def __init__(self, in_channels=1, out_channels=1, base_filters=32):
        super().__init__()
        # Encoder
        self.enc1 = ResidualBlock(in_channels, base_filters)
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = ResidualBlock(base_filters, base_filters*2)
        self.pool2 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = ResidualBlock(base_filters*2, base_filters*4)
        
        # Decoder
        self.up2 = nn.ConvTranspose2d(base_filters*4, base_filters*2, 2, stride=2)
        self.dec2 = ResidualBlock(base_filters*4, base_filters*2)
        
        self.up1 = nn.ConvTranspose2d(base_filters*2, base_filters, 2, stride=2)
        self.dec1 = ResidualBlock(base_filters*2, base_filters)
        
        # Output
        self.out = nn.Conv2d(base_filters, out_channels, 1)
        
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        
        # Bottleneck
        b = self.bottleneck(self.pool2(e2))
        
        # Decoder
        d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        
        return self.out(d1)

# %%


# %%
model = UNetResidualLite(in_channels=1, out_channels=1, base_filters=2).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
helper.inspect_model(
    model, 
    input_size=(1,1,HEIGHT,WIDTH),
    criterion=criterion,
    optimizer=optimizer,
    model_name="U-Net-Lite with Residual Connections"
)

# %%
"""
SSIM (Structural Similarity Index) Ranges:
- 0.95 - 1.00: Excellent quality (near perfect reconstruction)
- 0.90 - 0.95: Very good quality (minor differences)
- 0.80 - 0.90: Good quality (noticeable but acceptable differences)
- 0.70 - 0.80: Fair quality (visible degradation)
- 0.60 - 0.70: Poor quality (significant artifacts)
- < 0.60: Very poor quality

PSNR (Peak Signal-to-Noise Ratio) Ranges:
- 40 dB: Excellent quality (near perfect)
- 30 - 40 dB: Good to very good quality
- 20 - 30 dB: Acceptable to fair quality
- < 20 dB: Poor quality
"""

# %%
from torchmetrics import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    total_ssim = 0
    total_psnr = 0
    
    # Initialize metrics
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    
    for imgs, masks in tqdm(loader, desc="Training", leave=False):
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        
        loss = criterion(outputs, masks)  
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Calculate metrics
        total_ssim += ssim_metric(outputs, masks).item()
        total_psnr += psnr_metric(outputs, masks).item()
    
    n = len(loader)
    return total_loss / n, total_ssim / n, total_psnr / n

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    total_ssim = 0
    total_psnr = 0
    
    # Initialize metrics
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    
    with torch.no_grad():
        for imgs, masks in tqdm(loader, desc="Validation", leave=False):
            imgs, masks = imgs.to(device), masks.to(device)
            outputs = model(imgs)
            
            loss = criterion(outputs, masks)  
            total_loss += loss.item()
            
            # Calculate metrics
            total_ssim += ssim_metric(outputs, masks).item()
            total_psnr += psnr_metric(outputs, masks).item()
    
    n = len(loader)
    return total_loss / n, total_ssim / n, total_psnr / n


# %%
"""
https://medium.com/data-science/super-resolution-a-basic-study-e01af1449e13 - Total Variation Loss , SRCNN
"""

# %%
import torch
import torch.nn as nn

class TotalVariationLoss(nn.Module):
    def __init__(self, weight=1.0):
        """
        weight: Scaling factor for the TV loss
        """
        super(TotalVariationLoss, self).__init__()
        self.weight = weight

    def forward(self, predicted , true):
        """
        predicted: input image tensor of shape (B, C, H, W)
        returns: scalar TV loss
        """
        # Difference between neighboring pixels along height
        diff_h = predicted[:, :, 1:, :] - predicted[:, :, :-1, :]
        # Difference between neighboring pipredictedels along width
        diff_w = predicted[:, :, :, 1:] - predicted[:, :, :, :-1]
        # Sum of squares (L2) or absolute values (L1) can be used
        loss = torch.sum(torch.abs(diff_h)) + torch.sum(torch.abs(diff_w))
        # Normalize by batch size
        loss = self.weight * loss / predicted.size(0)
        return loss
    



# %%


# %%
CRITERIONS = {
    # "TotalVariationLoss" : TotalVariationLoss() , Utter Flop
    "L1Loss" : nn.L1Loss()
}
HISTORY = {
    
}

# %%
for name, criterion in CRITERIONS.items():
    print("Using : ", name)
    model = UNetResidualLite(in_channels=1, out_channels=1, base_filters=2).to(device)
    # Training with detailed loss tracking
    epochs = 25
    history = {
        'train_loss': [], 
        'val_loss': [],
        'train_ssim': [],
        'val_ssim': [],
        'train_psnr': [],
        'val_psnr': []
    }
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        train_loss, train_ssim_acc, train_psnr_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_ssim_acc, val_psnr_acc = validate(model, val_loader, criterion, device)
        
        # Store history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_ssim'].append(train_ssim_acc)
        history['val_ssim'].append(val_ssim_acc)
        history['train_psnr'].append(train_psnr_acc)
        history['val_psnr'].append(val_psnr_acc)
        
        scheduler.step(val_loss)
        
        # Compact printing format
        print(f"Epoch {epoch+1:2d}/{epochs} | "
              f"Loss: {train_loss:.4f}/{val_loss:.4f} | "
              f"SSIM: {train_ssim_acc:.3f}/{val_ssim_acc:.3f} | "
              f"PSNR: {train_psnr_acc:5.1f}/{val_psnr_acc:5.1f} dB", end="")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_OUTPUT_DIR / f"best_model_{name}.pth")
            print(" *", end="")
        print()
    
    HISTORY[name] = history
    
    # Final summary
    best_val_loss_epoch = np.argmin(history['val_loss']) + 1
    best_val_ssim = max(history['val_ssim'])
    best_val_ssim_epoch = np.argmax(history['val_ssim']) + 1
    best_val_psnr = max(history['val_psnr'])
    best_val_psnr_epoch = np.argmax(history['val_psnr']) + 1
    
    print(f"\n{name} Summary:")
    print(f"  Best Val Loss:  {min(history['val_loss']):.4f} (Epoch {best_val_loss_epoch})")
    print(f"  Best Val SSIM:  {best_val_ssim:.4f} (Epoch {best_val_ssim_epoch})")
    print(f"  Best Val PSNR:  {best_val_psnr:.2f} dB (Epoch {best_val_psnr_epoch})")

# %%
! ipynb-py-convert 2-residual-block-loss.ipynb main.py

# %%
# Plot for each criterion in HISTORY using subplot
for criterion_name, history in HISTORY.items():
    plt.figure(figsize=(18, 4))
    
    # 1st subplot: Loss
    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='Training Loss', color='blue', linewidth=2)
    plt.plot(history['val_loss'], label='Validation Loss', color='red', linewidth=2)
    plt.title(f'{criterion_name} - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 2nd subplot: SSIM
    plt.subplot(1, 3, 2)
    plt.plot(history['train_ssim'], label='Training SSIM', color='green', linewidth=2)
    plt.plot(history['val_ssim'], label='Validation SSIM', color='orange', linewidth=2)
    plt.title(f'{criterion_name} - SSIM')
    plt.xlabel('Epoch')
    plt.ylabel('SSIM')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 3rd subplot: PSNR
    plt.subplot(1, 3, 3)
    plt.plot(history['train_psnr'], label='Training PSNR', color='purple', linewidth=2)
    plt.plot(history['val_psnr'], label='Validation PSNR', color='brown', linewidth=2)
    plt.title(f'{criterion_name} - PSNR')
    plt.xlabel('Epoch')
    plt.ylabel('PSNR (dB)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Print final values
    print(f"{criterion_name} - Final Metrics:")
    print(f"  Loss:  Train={history['train_loss'][-1]:.4f}, Val={history['val_loss'][-1]:.4f}")
    print(f"  SSIM:  Train={history['train_ssim'][-1]:.4f}, Val={history['val_ssim'][-1]:.4f}")
    print(f"  PSNR:  Train={history['train_psnr'][-1]:.2f} dB, Val={history['val_psnr'][-1]:.2f} dB")
    print("-" * 60)

# %%


# %%
# Load best model
model.load_state_dict(torch.load(MODEL_OUTPUT_DIR / "best_model.pth"))
model.eval()
# Get random images
input_path, target_path = helper.get_images_from_dirs([INPUT_DIR, TARGET_DIR])
# Load and preprocess
img_pil = Image.open(input_path).convert('L')
mask_pil = Image.open(target_path).convert('L')

img_resized = img_pil.resize((HEIGHT, WIDTH))
img_tensor = torch.from_numpy(np.array(img_resized).astype(np.float32) / 255.0).unsqueeze(0).to(device)
img_tensor = img_tensor.unsqueeze(0)  # This makes it (1, 1, H, W)

# Predict
with torch.no_grad():
    pred = model(img_tensor).cpu().numpy()

print(f"Original pred shape: {pred.shape}")
pred = pred.squeeze() # This converts (1, 1, 256, 256) -> (256, 256)
print(f"After squeeze shape: {pred.shape}")

# Convert for visualization
img_np = np.array(img_resized)
mask_np = np.array(mask_pil.resize((HEIGHT, WIDTH))) / 255.0

# Display using helper
helper.show_grid({
    'Noise Mask': (img_np * 255).astype(np.uint8),
    'Target Mask': (mask_np * 255).astype(np.uint8),
    'Predicted Mask': (pred * 255).astype(np.uint8),  # Now this is 2D
}, grid="row", width=1200)

# %%
