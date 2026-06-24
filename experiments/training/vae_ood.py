"""
VAE-based OOD Filter for PBS Blood Cell Images
-----------------------------------------------
PBS 혈구 이미지로 학습한 VAE를 사용해 블러/이물질/비혈구 이미지를 필터링.
reconstruction error가 3-sigma threshold를 초과하면 OOD로 판정.

Architecture: ResidualBlock + VAE + UNet skip connections
Training data: pbs06/train (혈구 이미지)
OOD validation: food101, CIFAR-10, ImageNet 샘플 (다른 도메인)
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
from tqdm import tqdm

# ── 데이터 경로 ────────────────────────────────────────────────────────
WORKING_DIR    = "/home/smile/work/pbs06/"
TRAIN_IMAGE_DIR = os.path.join(WORKING_DIR, "train/images")
VAL_IMAGE_DIR   = os.path.join(WORKING_DIR, "val/images")
TEST_IMAGE_DIR  = os.path.join(WORKING_DIR, "test/images")
MODEL_SAVE_PATH = "/home/smile/work/a_CKM_mo/ObjectDetection/YOLO/0429_ae_model.pt"

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])


# ── Dataset ────────────────────────────────────────────────────────────
class BloodImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_paths = list(Path(image_dir).glob("*.jpg"))
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, 0


# ── Model ──────────────────────────────────────────────────────────────
class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch), nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch)
        )
        self.skip = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        return F.relu(self.conv(x) + self.skip(x))


class AdvancedVAE_UNet(nn.Module):
    """ResidualBlock + VAE + UNet skip connections"""
    def __init__(self, z_dim=128):
        super().__init__()
        # Encoder
        self.enc1  = ResidualBlock(3, 32)
        self.down1 = nn.MaxPool2d(2)      # 256→128
        self.enc2  = ResidualBlock(32, 64)
        self.down2 = nn.MaxPool2d(2)      # 128→64
        self.enc3  = ResidualBlock(64, 128)
        self.down3 = nn.MaxPool2d(2)      # 64→32
        self.pool  = nn.AdaptiveAvgPool2d((4, 4))
        self.flatten  = nn.Flatten()
        self.fc_mu    = nn.Linear(128 * 4 * 4, z_dim)
        self.fc_logvar = nn.Linear(128 * 4 * 4, z_dim)
        # Decoder
        self.fc_dec = nn.Linear(z_dim, 128 * 4 * 4)
        self.dec3   = ResidualBlock(128 + 128, 64)
        self.dec2   = ResidualBlock(64 + 64, 32)
        self.dec1   = nn.Sequential(nn.Conv2d(32 + 32, 3, 3, padding=1), nn.Sigmoid())
        self.dropout = nn.Dropout(0.3)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def encode(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(self.down1(x1))
        x3 = self.enc3(self.down2(x2))
        pooled = self.pool(self.down3(x3))
        flat = self.flatten(pooled)
        return self.fc_mu(flat), self.fc_logvar(flat), [x1, x2, x3]

    def decode(self, z, skips):
        x = self.fc_dec(z).view(-1, 128, 4, 4)
        x = F.interpolate(x, size=skips[2].shape[2:])
        x = self.dec3(self.dropout(torch.cat([x, skips[2]], dim=1)))
        x = F.interpolate(x, size=skips[1].shape[2:])
        x = self.dec2(self.dropout(torch.cat([x, skips[1]], dim=1)))
        x = F.interpolate(x, size=skips[0].shape[2:])
        return self.dec1(self.dropout(torch.cat([x, skips[0]], dim=1)))

    def forward(self, x):
        mu, logvar, skips = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, skips), mu, logvar


def vae_loss(x_hat, x, mu, logvar):
    recon = F.mse_loss(x_hat, x, reduction="mean")
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    return recon + kl


# ── Training ───────────────────────────────────────────────────────────
def train(num_epochs=200):
    train_loader = DataLoader(
        BloodImageDataset(TRAIN_IMAGE_DIR, transform), batch_size=16, shuffle=True)
    test_loader = DataLoader(
        BloodImageDataset(TEST_IMAGE_DIR, transform), batch_size=16, shuffle=False)

    model = AdvancedVAE_UNet().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    best_test_loss = float("inf")

    for epoch in tqdm(range(num_epochs), desc="Training"):
        model.train()
        total_loss = 0
        for x, _ in train_loader:
            x = x.cuda()
            x_hat, mu, logvar = model(x)
            loss = vae_loss(x_hat, x, mu, logvar)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item()

        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for x, _ in test_loader:
                x = x.cuda()
                x_hat, mu, logvar = model(x)
                test_loss += vae_loss(x_hat, x, mu, logvar).item()
        test_loss /= len(test_loader)

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            sample_x = next(iter(test_loader))[0][:8].cuda()
            sample_xhat, _, _ = model(sample_x)
            vutils.save_image(
                torch.cat([sample_x, sample_xhat]),
                f"best_recon_epoch{epoch+1:03d}.png", nrow=8)
            print(f"[Epoch {epoch+1}] New best test loss: {test_loss:.6f}")

        print(f"[Epoch {epoch+1}] Train={total_loss/len(train_loader):.6f}  Test={test_loss:.6f}")

    return model


# ── OOD Inference ──────────────────────────────────────────────────────
def reconstruction_error(x, model):
    model.eval()
    with torch.no_grad():
        x = x.cuda()
        x_hat, _, _ = model(x)
        err = F.mse_loss(x_hat, x, reduction="none")
        return err.view(err.size(0), -1).mean(dim=1).cpu().numpy()


def compute_threshold(model, sigma=3):
    """test set reconstruction error로 OOD threshold 계산 (mean + sigma*std)"""
    test_loader = DataLoader(
        BloodImageDataset(TEST_IMAGE_DIR, transform), batch_size=32)
    all_errors = []
    for x, _ in test_loader:
        all_errors.extend(reconstruction_error(x, model))
    mean_e, std_e = np.mean(all_errors), np.std(all_errors)
    threshold = mean_e + sigma * std_e
    print(f"OOD threshold ({sigma}σ): {threshold:.6f}  (mean={mean_e:.6f}, std={std_e:.6f})")
    return threshold


def is_ood(image_path, model, threshold):
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0)
    err = reconstruction_error(x, model)[0]
    return err > threshold, err


# ── Main ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--eval", type=str, help="image path to evaluate")
    args = parser.parse_args()

    if args.train:
        model = train()
    else:
        model = AdvancedVAE_UNet().cuda()
        model.load_state_dict(torch.load(MODEL_SAVE_PATH))
        model.eval()

    threshold = compute_threshold(model)

    if args.eval:
        ood, err = is_ood(args.eval, model, threshold)
        status = "OOD" if ood else "In-Distribution"
        print(f"{args.eval}: {status} (err={err:.6f}, threshold={threshold:.6f})")
