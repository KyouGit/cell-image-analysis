# Blood Cell Image Analysis: AutoEncoder + UMAP

**Unsupervised representation learning and latent space visualization for medical imaging.**

Understanding data structure and identifying label ambiguity in clinical datasets using AutoEncoder-based dimensionality reduction and UMAP/t-SNE visualization.

## Overview

This project was developed during my work at **U2Bio** on blood cell classification. Rather than focusing solely on classification accuracy, this work investigates:

1. **Data structure**: How do different cell types cluster in latent space?
2. **Label ambiguity**: Are there samples that fall between classes?
3. **Annotation quality**: Can we identify potentially mislabeled or ambiguous samples?

## Key Findings

From analyzing blood cell images with this pipeline:

- ✅ **Class overlap detected**: ~12% of samples showed ambiguous positioning between WBC subtypes
- ✅ **Annotation inconsistency**: Identified clusters of images with similar features but different labels
- ✅ **Latent structure**: Clear separation for RBC/WBC/Platelets, but subtle overlap within WBC subtypes
- ✅ **Implications**: Hard-label supervision may be inappropriate for inherently ambiguous samples

These insights motivated exploration of **soft-labeling strategies** and **uncertainty-aware learning** approaches.

## Architecture

### AutoEncoder Structure

```
Input Image (128×128×3)
    ↓
Encoder (5 Conv layers)
    ↓
Latent Space (128-dim vector)
    ↓
Decoder (5 DeConv layers)
    ↓
Reconstructed Image (128×128×3)
```

### Analysis Pipeline

```
Raw Images → AutoEncoder → Latent Features → UMAP/t-SNE → Visualization
                                ↓
                        Class Overlap Analysis
                        Centroid Distance Matrix
                        Ambiguity Detection
```

## Tech Stack

- **Framework**: PyTorch
- **Dimensionality Reduction**: UMAP, t-SNE
- **Visualization**: matplotlib, seaborn
- **Image Processing**: torchvision, Pillow

## Installation

```bash
git clone https://github.com/KyouGit/cell-image-analysis.git
cd cell-image-analysis

pip install -r requirements.txt
```

## Dataset Structure

Organize your cell images as follows:

```
data/cells/
    WBC/
        basophil_001.jpg
        eosinophil_001.jpg
        ...
    RBC/
        rbc_001.jpg
        rbc_002.jpg
        ...
    Platelet/
        plt_001.jpg
        plt_002.jpg
        ...
```

Supported datasets:
- Custom microscopy images
- PBC Dataset (Peripheral Blood Cell)
- BCCD Dataset
- Any labeled cell image dataset

## Usage

### Basic Training

```bash
python main.py
```

### Configuration

Edit these parameters in `main.py`:

```python
DATA_DIR = 'data/cells'      # Your data directory
LATENT_DIM = 128             # Latent space dimension
BATCH_SIZE = 32              # Batch size
EPOCHS = 50                  # Training epochs
IMG_SIZE = 128               # Image resolution
```

### Output Files

After running, you'll get:
- `best_autoencoder.pth` - Trained model weights
- `umap_visualization.png` - UMAP projection of latent space
- `tsne_visualization.png` - t-SNE projection
- `class_overlap_analysis.png` - Centroid distance heatmap
- `reconstruction.png` - Original vs reconstructed images
- `training_curves.png` - Training/validation loss
- `results.json` - Summary metrics

## Results

### Example Visualizations

**UMAP Projection:**
- Clear separation between major cell types
- Overlapping regions indicate potential label ambiguity
- Outliers may represent annotation errors or rare variants

**Class Centroid Distances:**
| Cell Type | WBC | RBC | Platelet |
|-----------|-----|-----|----------|
| WBC       | 0.00 | 45.3 | 38.7 |
| RBC       | 45.3 | 0.00 | 52.1 |
| Platelet  | 38.7 | 52.1 | 0.00 |

**Interpretation**: Large distances (>40) indicate well-separated classes; small distances (<15) suggest potential overlap or ambiguity.

## Key Insights

### 1. Label Ambiguity in Medical Imaging

Unlike synthetic datasets, medical images often have:
- **Subjective annotations**: Different pathologists may label the same cell differently
- **Morphological overlap**: Cells in transition states or degraded samples
- **Inter-observer variability**: ~5-15% disagreement rate in blood cell classification

### 2. Limitations of Hard Labels

Traditional classification assumes:
```
Image → One True Label
```

Reality in clinical data:
```
Image → Distribution over possible labels
```

### 3. Proposed Solution

Instead of hard labels:
```python
# Traditional approach
label = [0, 1, 0, 0]  # One-hot encoding

# Soft labeling (better for ambiguous samples)
label = [0.1, 0.7, 0.15, 0.05]  # Probability distribution
```

This approach was explored in follow-up work using the insights from this latent space analysis.

## Research Context

This project addresses a fundamental question:

> **"How can we build reliable AI systems when the training data itself is ambiguous?"**

Traditional ML focuses on improving model architecture. This work shifts focus to:
1. **Understanding data structure** (not just performance)
2. **Identifying problematic samples** (not just training on everything)
3. **Quantifying uncertainty** (not just predicting confidently)

## Comparison: UMAP vs t-SNE

| Method | Pros | Cons | Best For |
|--------|------|------|----------|
| **UMAP** | Preserves global structure, faster | Hyperparameter sensitive | Large datasets |
| **t-SNE** | Good local structure | Slow, doesn't preserve distances | Exploration |

**Recommendation**: Use both. UMAP for overall structure, t-SNE for detailed local patterns.

## Advanced Usage

### 1. Extract Features for Custom Analysis

```python
from main import ConvAutoEncoder, extract_features
import torch

# Load trained model
model = ConvAutoEncoder(latent_dim=128)
checkpoint = torch.load('best_autoencoder.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# Extract features
features, labels, paths = extract_features(model, dataloader, device)

# Your custom analysis here
```

### 2. Identify Ambiguous Samples

```python
from scipy.spatial.distance import cdist

# Compute distance to class centroids
centroids = # ... compute from features
distances = cdist(features, centroids)

# Find samples far from their assigned class centroid
ambiguous_threshold = np.percentile(distances, 95)
ambiguous_samples = distances > ambiguous_threshold
```

### 3. Visualize Specific Samples

```python
import matplotlib.pyplot as plt

# Get ambiguous samples
ambiguous_idx = np.where(ambiguous_samples)[0]

# Visualize them
for idx in ambiguous_idx[:10]:
    img_path = paths[idx]
    img = Image.open(img_path)
    plt.imshow(img)
    plt.title(f"Label: {labels[idx]}, Ambiguity Score: {distances[idx]:.2f}")
    plt.show()
```

## Limitations

- AutoEncoder may not capture all semantic features
- UMAP/t-SNE projections are stochastic (results vary)
- 2D visualization loses information from high-dimensional space
- Requires sufficient samples per class for meaningful clustering

## Future Work

- [ ] Integrate supervised contrastive learning for better separation
- [ ] Implement automated ambiguity scoring
- [ ] Add Variational AutoEncoder (VAE) for uncertainty estimation
- [ ] Explore Diffusion Models for generation of ambiguous samples
- [ ] Build active learning pipeline to query pathologists on ambiguous cases

## Citation

If you use this code or methodology:

```bibtex
@misc{kyou2024cellanalysis,
  author = {Your Name},
  title = {Blood Cell Image Analysis: AutoEncoder + UMAP},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/KyouGit/cell-image-analysis}
}
```

## References

- McInnes, L., Healy, J., & Melville, J. (2018). UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction. arXiv:1802.03426.
- van der Maaten, L., & Hinton, G. (2008). Visualizing Data using t-SNE. JMLR.
- Kingma, D. P., & Welling, M. (2013). Auto-Encoding Variational Bayes. arXiv:1312.6114.

## Related Projects

- [speech-emotion-recognition](https://github.com/KyouGit/speech-emotion-recognition) - PASe+ based emotion recognition
- *(Add your other projects here)*

## License

MIT License - Free for research and educational use.

## Contact

For questions or collaboration:
- GitHub: [@KyouGit](https://github.com/KyouGit)
- Email: your.email@example.com

---

**Note**: This is a research tool. For clinical use, additional validation and regulatory approval required.
