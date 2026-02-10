"""
Simple example: Feature extraction and visualization

This script demonstrates how to use a pretrained AutoEncoder
to extract features and visualize them.
"""

import torch
import numpy as np
from main import ConvAutoEncoder, CellImageDataset, extract_features
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import umap


def quick_analysis(data_dir, model_path='best_autoencoder.pth'):
    """
    Quick latent space analysis
    
    Args:
        data_dir: Path to image directory
        model_path: Path to trained model
    """
    print("="*60)
    print("Quick Latent Space Analysis")
    print("="*60)
    
    # Load dataset
    print("\n[1/4] Loading dataset...")
    dataset = CellImageDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    # Load model
    print("\n[2/4] Loading model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ConvAutoEncoder(latent_dim=128)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Extract features
    print("\n[3/4] Extracting features...")
    features, labels, paths = extract_features(model, dataloader, device)
    print(f"Feature shape: {features.shape}")
    
    # Visualize
    print("\n[4/4] Creating visualization...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    features_2d = reducer.fit_transform(features)
    
    plt.figure(figsize=(12, 10))
    
    unique_labels = np.unique(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    
    for label, color in zip(unique_labels, colors):
        mask = labels == label
        class_name = dataset.idx_to_class[label]
        plt.scatter(features_2d[mask, 0], features_2d[mask, 1],
                   c=[color], label=class_name,
                   alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    
    plt.title('Latent Space (UMAP)', fontsize=16, fontweight='bold')
    plt.xlabel('UMAP 1', fontsize=12)
    plt.ylabel('UMAP 2', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('quick_analysis.png', dpi=150)
    print("\nSaved: quick_analysis.png")
    print("="*60)


def find_similar_images(data_dir, query_image_path, model_path='best_autoencoder.pth', top_k=5):
    """
    Find images similar to a query image using latent space distance
    
    Args:
        data_dir: Path to image directory
        query_image_path: Path to query image
        model_path: Path to trained model
        top_k: Number of similar images to return
    """
    from PIL import Image
    from torchvision import transforms
    from scipy.spatial.distance import cdist
    
    print(f"\n[INFO] Finding images similar to: {query_image_path}")
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ConvAutoEncoder(latent_dim=128)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Load and encode query image
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    query_img = Image.open(query_image_path).convert('RGB')
    query_tensor = transform(query_img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        _, query_features = model(query_tensor)
        query_features = query_features.cpu().numpy()
    
    # Load dataset and extract all features
    dataset = CellImageDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    all_features, all_labels, all_paths = extract_features(model, dataloader, device)
    
    # Compute distances
    distances = cdist(query_features, all_features, metric='euclidean')[0]
    
    # Get top-k similar images
    top_indices = np.argsort(distances)[:top_k]
    
    # Display results
    print(f"\nTop {top_k} similar images:")
    print("-" * 60)
    for i, idx in enumerate(top_indices):
        print(f"{i+1}. {all_paths[idx]}")
        print(f"   Distance: {distances[idx]:.4f}, Label: {dataset.idx_to_class[all_labels[idx]]}")
    print("-" * 60)
    
    # Visualize
    fig, axes = plt.subplots(1, top_k + 1, figsize=(top_k * 2 + 2, 2))
    
    # Query image
    axes[0].imshow(query_img)
    axes[0].set_title('Query', fontweight='bold')
    axes[0].axis('off')
    
    # Similar images
    for i, idx in enumerate(top_indices):
        img = Image.open(all_paths[idx])
        axes[i+1].imshow(img)
        axes[i+1].set_title(f'#{i+1}\nDist: {distances[idx]:.2f}')
        axes[i+1].axis('off')
    
    plt.tight_layout()
    plt.savefig('similar_images.png', dpi=150)
    print("\nSaved: similar_images.png")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Quick analysis:    python example.py data/cells/")
        print("  Find similar:      python example.py data/cells/ query_image.jpg")
        sys.exit(1)
    
    data_dir = sys.argv[1]
    
    if len(sys.argv) == 2:
        # Quick analysis
        quick_analysis(data_dir)
    else:
        # Find similar images
        query_path = sys.argv[2]
        find_similar_images(data_dir, query_path)
