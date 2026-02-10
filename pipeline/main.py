"""
Blood Cell Image Analysis: AutoEncoder + UMAP Visualization

This project demonstrates:
1. Learning latent representations of blood cell images using AutoEncoder
2. Visualizing the learned representations with UMAP/t-SNE
3. Identifying label ambiguity and class overlap in medical imaging datasets

Developed as part of research at U2Bio on understanding data structure
and annotation quality in clinical datasets.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
import umap
from tqdm import tqdm
import json
import warnings
warnings.filterwarnings('ignore')


def get_train_transform(img_size=128):
    """Training transform with data augmentation and [-1, 1] normalization"""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])


def get_val_transform(img_size=128):
    """Validation/test transform with [-1, 1] normalization (no augmentation)"""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])


def denormalize(tensor):
    """Denormalize from [-1, 1] back to [0, 1] for visualization"""
    return torch.clamp(tensor * 0.5 + 0.5, 0, 1)


class CellImageDataset(Dataset):
    """
    Dataset for cell images

    Expected structure:
        data/
            class1/
                img1.jpg
                img2.jpg
            class2/
                img1.jpg
                img2.jpg
    """
    def __init__(self, data_dir, transform=None, img_size=128):
        self.data_dir = Path(data_dir)
        self.img_size = img_size

        if transform is None:
            self.transform = get_val_transform(img_size)
        else:
            self.transform = transform

        # Load all images
        self.images = []
        self.labels = []
        self.class_to_idx = {}
        self.idx_to_class = {}

        class_folders = sorted([d for d in self.data_dir.iterdir() if d.is_dir()])

        for idx, class_folder in enumerate(class_folders):
            class_name = class_folder.name
            self.class_to_idx[class_name] = idx
            self.idx_to_class[idx] = class_name

            # Support multiple image formats
            img_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                img_files.extend(list(class_folder.glob(ext)))

            self.images.extend(img_files)
            self.labels.extend([idx] * len(img_files))

        print(f"[INFO] Loaded {len(self.images)} images from {len(self.class_to_idx)} classes")
        print(f"[INFO] Classes: {self.class_to_idx}")

        # Print class distribution
        unique, counts = np.unique(self.labels, return_counts=True)
        for cls_idx, count in zip(unique, counts):
            print(f"  {self.idx_to_class[cls_idx]}: {count} images")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]

        # Load image
        image = Image.open(img_path).convert('RGB')

        # Apply transform
        if self.transform:
            image = self.transform(image)

        return image, label, str(img_path)


class ResidualBlock(nn.Module):
    """Residual block: Conv-BN-LeakyReLU-Conv-BN + Skip Connection"""
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
        )
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.activation(self.block(x) + x)


class ConvAutoEncoder(nn.Module):
    """
    Convolutional AutoEncoder with Residual Blocks

    Architecture:
        Encoder: Conv+Residual layers → Bottleneck (latent representation)
        Decoder: DeConv+Residual layers → Reconstructed image
    """
    def __init__(self, latent_dim=256, img_channels=3):
        super().__init__()

        self.latent_dim = latent_dim

        # Encoder: (3, 128, 128) → (512, 4, 4) → latent_dim
        self.encoder = nn.Sequential(
            # (3, 128, 128) → (64, 64, 64)
            nn.Conv2d(img_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            ResidualBlock(64),

            # (64, 64, 64) → (128, 32, 32)
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            ResidualBlock(128),

            # (128, 32, 32) → (256, 16, 16)
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            ResidualBlock(256),

            # (256, 16, 16) → (512, 8, 8)
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            ResidualBlock(512),

            # (512, 8, 8) → (512, 4, 4)
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Latent space
        self.fc_encoder = nn.Linear(512 * 4 * 4, latent_dim)
        self.fc_decoder = nn.Linear(latent_dim, 512 * 4 * 4)

        # Decoder: latent_dim → (512, 4, 4) → (3, 128, 128)
        self.decoder = nn.Sequential(
            # (512, 4, 4) → (512, 8, 8)
            nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            ResidualBlock(512),

            # (512, 8, 8) → (256, 16, 16)
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            ResidualBlock(256),

            # (256, 16, 16) → (128, 32, 32)
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            ResidualBlock(128),

            # (128, 32, 32) → (64, 64, 64)
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            ResidualBlock(64),

            # (64, 64, 64) → (3, 128, 128)
            nn.ConvTranspose2d(64, img_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def encode(self, x):
        """Encode image to latent representation"""
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        z = self.fc_encoder(x)
        return z

    def decode(self, z):
        """Decode latent representation to image"""
        x = self.fc_decoder(z)
        x = x.view(x.size(0), 512, 4, 4)
        x = self.decoder(x)
        return x

    def forward(self, x):
        """Full forward pass: encode → decode"""
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z


class AutoEncoderTrainer:
    """Trainer for AutoEncoder with CosineAnnealing scheduler"""

    def __init__(self, model, device=None, lr=5e-4, epochs=50):
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)

        self.history = {
            'train_loss': [],
            'val_loss': []
        }

    def train_epoch(self, dataloader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0

        for images, _, _ in tqdm(dataloader, desc="Training"):
            images = images.to(self.device)

            # Forward pass
            recon_images, _ = self.model(images)
            loss = self.criterion(recon_images, images)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        return avg_loss

    def validate(self, dataloader):
        """Validate the model"""
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for images, _, _ in dataloader:
                images = images.to(self.device)
                recon_images, _ = self.model(images)
                loss = self.criterion(recon_images, images)
                total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        return avg_loss

    def train(self, train_loader, val_loader, epochs=50):
        """Full training loop"""
        print(f"\n[INFO] Training AutoEncoder on {self.device}")
        print(f"[INFO] Latent dimension: {self.model.latent_dim}")
        print(f"[INFO] Learning rate: {self.optimizer.param_groups[0]['lr']}")

        best_val_loss = float('inf')

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            self.scheduler.step()

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)

            lr = self.optimizer.param_groups[0]['lr']
            print(f"Epoch [{epoch+1}/{epochs}] "
                  f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {lr:.6f}")

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model('outputs/best_autoencoder.pth')

        print(f"\n[INFO] Training completed. Best Val Loss: {best_val_loss:.4f}")

    def save_model(self, path):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history
        }, path)

    def load_model(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']


def extract_features(model, dataloader, device):
    """Extract latent features from all images"""
    model.eval()
    
    all_features = []
    all_labels = []
    all_paths = []
    
    print("[INFO] Extracting latent features...")
    
    with torch.no_grad():
        for images, labels, paths in tqdm(dataloader):
            images = images.to(device)
            _, features = model(images)
            
            all_features.append(features.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_paths.extend(paths)
    
    all_features = np.vstack(all_features)
    all_labels = np.array(all_labels)
    
    return all_features, all_labels, all_paths


def visualize_latent_space(features, labels, class_names, method='umap', save_path='latent_space.png', max_per_class=None):
    """
    Visualize latent space using UMAP or t-SNE

    Args:
        features: (N, latent_dim) array
        labels: (N,) array
        class_names: dict mapping label to class name
        method: 'umap' or 'tsne'
        max_per_class: Maximum samples per class for balanced visualization (None = no cap)
    """
    print(f"[INFO] Visualizing latent space with {method.upper()}...")

    # Subsample for class balance if requested
    if max_per_class is not None:
        selected_idx = []
        rng = np.random.RandomState(42)
        for label in np.unique(labels):
            class_idx = np.where(labels == label)[0]
            if len(class_idx) > max_per_class:
                class_idx = rng.choice(class_idx, max_per_class, replace=False)
            selected_idx.extend(class_idx)
        selected_idx = np.array(sorted(selected_idx))
        features = features[selected_idx]
        labels = labels[selected_idx]
        print(f"[INFO] Subsampled to {len(features)} samples (max {max_per_class}/class)")

    # Dimensionality reduction
    if method == 'umap':
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    elif method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
    else:
        raise ValueError(f"Unknown method: {method}")

    features_2d = reducer.fit_transform(features)
    
    # Plot
    plt.figure(figsize=(12, 10))
    
    unique_labels = np.unique(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    
    for label, color in zip(unique_labels, colors):
        mask = labels == label
        plt.scatter(features_2d[mask, 0], features_2d[mask, 1],
                   c=[color], label=class_names[label],
                   alpha=0.6, s=30, edgecolors='black', linewidth=0.5)
    
    plt.title(f'Latent Space Visualization ({method.upper()})', fontsize=16, fontweight='bold')
    plt.xlabel(f'{method.upper()} Component 1', fontsize=12)
    plt.ylabel(f'{method.upper()} Component 2', fontsize=12)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[INFO] Saved visualization to {save_path}")
    plt.close()
    
    return features_2d


def visualize_reconstruction(model, dataloader, device, num_samples=8, save_path='reconstruction.png'):
    """Visualize original vs reconstructed images"""
    model.eval()

    # Get one batch
    images, labels, _ = next(iter(dataloader))
    images = images[:num_samples].to(device)

    with torch.no_grad():
        recon_images, _ = model(images)

    images = denormalize(images.cpu())
    recon_images = denormalize(recon_images.cpu())
    
    # Plot
    fig, axes = plt.subplots(2, num_samples, figsize=(num_samples * 2, 4))
    
    for i in range(num_samples):
        # Original
        axes[0, i].imshow(images[i].permute(1, 2, 0).numpy())
        axes[0, i].set_title('Original', fontsize=10)
        axes[0, i].axis('off')
        
        # Reconstructed
        axes[1, i].imshow(recon_images[i].permute(1, 2, 0).numpy())
        axes[1, i].set_title('Reconstructed', fontsize=10)
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[INFO] Saved reconstruction visualization to {save_path}")
    plt.close()


def plot_training_curves(history, save_path='training_curves.png'):
    """Plot training and validation loss"""
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Train Loss', linewidth=2)
    plt.plot(history['val_loss'], label='Val Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Reconstruction Loss (MSE)', fontsize=12)
    plt.title('AutoEncoder Training Curves', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[INFO] Saved training curves to {save_path}")
    plt.close()


def analyze_class_overlap(features, labels, class_names, save_path='class_overlap_analysis.png'):
    """
    Analyze class overlap in latent space
    This helps identify label ambiguity
    """
    from scipy.spatial.distance import cdist
    
    print("[INFO] Analyzing class overlap...")
    
    unique_labels = np.unique(labels)
    n_classes = len(unique_labels)
    
    # Compute class centroids
    centroids = []
    for label in unique_labels:
        mask = labels == label
        centroid = features[mask].mean(axis=0)
        centroids.append(centroid)
    centroids = np.array(centroids)
    
    # Compute pairwise distances between centroids
    centroid_distances = cdist(centroids, centroids, metric='euclidean')
    
    # Plot heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(centroid_distances, annot=True, fmt='.2f', cmap='YlOrRd',
                xticklabels=[class_names[i] for i in unique_labels],
                yticklabels=[class_names[i] for i in unique_labels],
                cbar_kws={'label': 'Euclidean Distance'})
    plt.title('Class Centroid Distances in Latent Space', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[INFO] Saved class overlap analysis to {save_path}")
    plt.close()
    
    # Print insights
    print("\n[ANALYSIS] Class Separation:")
    min_dist_idx = np.unravel_index(np.argmin(centroid_distances + np.eye(n_classes) * 1e6), 
                                     centroid_distances.shape)
    max_dist_idx = np.unravel_index(np.argmax(centroid_distances), centroid_distances.shape)
    
    print(f"  Most similar classes: {class_names[min_dist_idx[0]]} ↔ {class_names[min_dist_idx[1]]} "
          f"(distance: {centroid_distances[min_dist_idx]:.2f})")
    print(f"  Most distinct classes: {class_names[max_dist_idx[0]]} ↔ {class_names[max_dist_idx[1]]} "
          f"(distance: {centroid_distances[max_dist_idx]:.2f})")


def main():
    """Main execution pipeline"""

    # Configuration
    DATA_DIR = 'data/cropped_cells'  # Cropped cells from YOLO detection
    OUTPUT_DIR = 'outputs'
    LATENT_DIM = 256
    BATCH_SIZE = 32
    EPOCHS = 50
    IMG_SIZE = 128
    LR = 5e-4
    MAX_PER_CLASS = 1000  # Cap samples per class for balanced visualization (None = no cap)

    print("="*70)
    print("Blood Cell Image Analysis: AutoEncoder + UMAP")
    print("="*70)

    # Create output directory
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # Check data directory
    if not Path(DATA_DIR).exists():
        print(f"\n[WARNING] Data directory not found: {DATA_DIR}")
        print("[INFO] Expected structure:")
        print("  data/cropped_cells/")
        print("      WBC/")
        print("          img1.jpg")
        print("      RBC/")
        print("          img1.jpg")
        print("      Platelet/")
        print("\nRun prepare_data.py then yolo_detect.py first.")
        return

    # 1. Load datasets with separate transforms
    print("\n[Step 1] Loading dataset...")
    train_transform = get_train_transform(IMG_SIZE)
    val_transform = get_val_transform(IMG_SIZE)

    # Load full dataset first to get class info and split indices
    full_dataset = CellImageDataset(DATA_DIR, transform=val_transform, img_size=IMG_SIZE)

    # 2. Split data
    print("\n[Step 2] Splitting data...")
    train_idx, val_idx = train_test_split(
        range(len(full_dataset)), test_size=0.2, random_state=42,
        stratify=full_dataset.labels
    )

    # Create separate datasets for train (with augmentation) and val (without)
    train_dataset_aug = CellImageDataset(DATA_DIR, transform=train_transform, img_size=IMG_SIZE)
    train_subset = torch.utils.data.Subset(train_dataset_aug, train_idx)
    val_subset = torch.utils.data.Subset(full_dataset, val_idx)

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    full_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"  Train: {len(train_subset)}, Val: {len(val_subset)}")

    # 3. Initialize model
    print("\n[Step 3] Initializing AutoEncoder...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ConvAutoEncoder(latent_dim=LATENT_DIM)
    trainer = AutoEncoderTrainer(model, device=device, lr=LR, epochs=EPOCHS)

    # 4. Train
    print("\n[Step 4] Training AutoEncoder...")
    trainer.train(train_loader, val_loader, epochs=EPOCHS)
    
    # 5. Extract features
    print("\n[Step 5] Extracting latent features...")
    features, labels, paths = extract_features(model, full_loader, device)
    print(f"  Feature shape: {features.shape}")
    
    # 6. Visualize with UMAP
    print("\n[Step 6] Visualizing latent space with UMAP...")
    visualize_latent_space(features, labels, full_dataset.idx_to_class,
                          method='umap', save_path=f'{OUTPUT_DIR}/umap_visualization.png',
                          max_per_class=MAX_PER_CLASS)

    # 7. Visualize with t-SNE
    print("\n[Step 7] Visualizing latent space with t-SNE...")
    visualize_latent_space(features, labels, full_dataset.idx_to_class,
                          method='tsne', save_path=f'{OUTPUT_DIR}/tsne_visualization.png',
                          max_per_class=MAX_PER_CLASS)

    # 8. Analyze class overlap
    print("\n[Step 8] Analyzing class overlap...")
    analyze_class_overlap(features, labels, full_dataset.idx_to_class,
                         save_path=f'{OUTPUT_DIR}/class_overlap_analysis.png')

    # 9. Visualize reconstruction
    print("\n[Step 9] Visualizing reconstruction quality...")
    visualize_reconstruction(model, val_loader, device,
                            save_path=f'{OUTPUT_DIR}/reconstruction.png')

    # 10. Plot training curves
    print("\n[Step 10] Plotting training curves...")
    plot_training_curves(trainer.history, save_path=f'{OUTPUT_DIR}/training_curves.png')
    
    # 11. Save results
    print("\n[Step 11] Saving results...")
    results = {
        'latent_dim': LATENT_DIM,
        'num_classes': len(full_dataset.class_to_idx),
        'class_mapping': full_dataset.class_to_idx,
        'num_images': len(full_dataset),
        'final_train_loss': trainer.history['train_loss'][-1],
        'final_val_loss': trainer.history['val_loss'][-1]
    }
    
    with open(f'{OUTPUT_DIR}/results.json', 'w') as f:
        json.dump(results, f, indent=4)

    print("\n" + "="*70)
    print("Analysis completed successfully!")
    print("="*70)
    print(f"\nOutput files (in {OUTPUT_DIR}/):")
    print("  - best_autoencoder.pth (model checkpoint)")
    print("  - umap_visualization.png (UMAP plot)")
    print("  - tsne_visualization.png (t-SNE plot)")
    print("  - class_overlap_analysis.png (centroid distances)")
    print("  - reconstruction.png (original vs reconstructed)")
    print("  - training_curves.png (loss curves)")
    print("  - results.json (summary)")
    print("="*70)


if __name__ == '__main__':
    main()
