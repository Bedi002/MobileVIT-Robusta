"""
05_train_unet_segmentasi.py
==============================
Melatih model segmentasi daun (U-Net ringan) SECARA OTOMATIS -- tanpa
bergantung pada anotasi polygon manual saat dipakai nanti. Anotasi polygon
cuma dipakai di sini untuk MELATIH segmenter-nya (sebagai ground truth
mask), persis seperti anotasi dipakai untuk melatih classifier sebelumnya.

PENTING soal kebocoran data: segmenter ini HANYA dilatih dari kelompok
pohon yang sama dengan train+valid classifier (bukan kelompok test/field).
Jadi baik classifier maupun segmenter sama-sama belum pernah "melihat"
kelompok pohon test -- pipeline deployment nanti (04b) betul-betul diuji
ke kondisi yang seluruhnya asing dari ujung ke ujung.

Cara pakai:
    python 05_train_unet_segmentasi.py \
        --csv "./Dataset/RoCoLe A robusta coffee leaf images dataset/Annotations/RoCoLE-csv.csv" \
        --photos_dir "./Dataset/RoCoLe A robusta coffee leaf images dataset/Photos" \
        --seed 42
"""

import os
import random
import argparse
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw
import numpy as np
from importlib import import_module

split_mod = import_module("01_split_domain_rocole")

IMG_SIZE = 256
BATCH_SIZE = 8
EPOCHS = 40
LR = 1e-3
SEED = 42


# ------------------------------------------------------------------
# U-Net ringan -- channel dikecilkan (16-32-64-128) supaya muat & cepat
# dilatih di GPU 6GB dengan dataset kecil.
# ------------------------------------------------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNetRingan(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, base=16):
        super().__init__()
        self.enc1 = ConvBlock(in_ch, base)
        self.enc2 = ConvBlock(base, base * 2)
        self.enc3 = ConvBlock(base * 2, base * 4)
        self.enc4 = ConvBlock(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(base * 8, base * 16)

        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)
        self.dec4 = ConvBlock(base * 16, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = ConvBlock(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = ConvBlock(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = ConvBlock(base * 2, base)

        self.out_conv = nn.Conv2d(base, out_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out_conv(d1)  # logits, belum di-sigmoid


class DiceBCELoss(nn.Module):
    def forward(self, logits, target):
        bce = nn.functional.binary_cross_entropy_with_logits(logits, target)
        probs = torch.sigmoid(logits)
        smooth = 1.0
        intersection = (probs * target).sum(dim=(1, 2, 3))
        dice = (2 * intersection + smooth) / (probs.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + smooth)
        dice_loss = 1 - dice.mean()
        return bce + dice_loss


class SegmentasiDataset(Dataset):
    """Dataset image + mask biner, dibangun on-the-fly dari polygon anotasi."""

    def __init__(self, entri_list, photos_dir, img_size=IMG_SIZE, augment=False):
        self.entri_list = entri_list
        self.photos_dir = photos_dir
        self.img_size = img_size
        self.augment = augment

    def __len__(self):
        return len(self.entri_list)

    def __getitem__(self, idx):
        entri = self.entri_list[idx]
        img_path = os.path.join(self.photos_dir, entri["filename"])
        img = Image.open(img_path).convert("RGB")

        mask = Image.new("L", img.size, 0)
        ImageDraw.Draw(mask).polygon(entri["polygon"], fill=255)

        img = img.resize((self.img_size, self.img_size))
        mask = mask.resize((self.img_size, self.img_size))

        if self.augment and random.random() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        img_arr = np.array(img, dtype=np.float32) / 255.0
        img_arr = (img_arr - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        img_tensor = torch.from_numpy(img_arr.transpose(2, 0, 1)).float()

        mask_arr = (np.array(mask, dtype=np.float32) / 255.0 > 0.5).astype(np.float32)
        mask_tensor = torch.from_numpy(mask_arr).unsqueeze(0).float()

        return img_tensor, mask_tensor


def siapkan_data_train_valid(csv_path, seed, rasio_train, rasio_valid):
    """Ambil HANYA entri dari kelompok pohon train+valid (bukan test/field) --
    persis mereproduksi split yang sama dengan 01_split_domain_rocole.py."""
    anotasi = split_mod.baca_anotasi(csv_path)
    per_pohon = defaultdict(list)
    for a in anotasi:
        kode = split_mod.ambil_kode_pohon(a["filename"])
        per_pohon[kode].append(a)

    kode_list = list(per_pohon.keys())
    random.Random(seed).shuffle(kode_list)
    n = len(kode_list)
    b1 = int(n * rasio_train)
    b2 = int(n * (rasio_train + rasio_valid))
    kode_train = set(kode_list[:b1])
    kode_valid = set(kode_list[b1:b2])

    entri_train = [e for k in kode_train for e in per_pohon[k]]
    entri_valid = [e for k in kode_valid for e in per_pohon[k]]
    return entri_train, entri_valid


def hitung_iou(logits, target, threshold=0.5):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    intersection = (preds * target).sum(dim=(1, 2, 3))
    union = ((preds + target) > 0).float().sum(dim=(1, 2, 3))
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--photos_dir", required=True)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--rasio_train", type=float, default=0.70)
    ap.add_argument("--rasio_valid", type=float, default=0.15)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    entri_train, entri_valid = siapkan_data_train_valid(
        args.csv, args.seed, args.rasio_train, args.rasio_valid
    )
    print(f"Data segmentasi -- train: {len(entri_train)}, valid: {len(entri_valid)}")
    print("(kelompok pohon test/field TIDAK disertakan sama sekali di sini)")

    train_ds = SegmentasiDataset(entri_train, args.photos_dir, augment=True)
    valid_ds = SegmentasiDataset(entri_valid, args.photos_dir, augment=False)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Perangkat aktif: {device}")

    model = UNetRingan().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Jumlah parameter U-Net: {n_params:,}")

    criterion = DiceBCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

    os.makedirs("checkpoints", exist_ok=True)
    best_iou = 0.0

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
        train_loss /= len(train_ds)

        model.eval()
        valid_loss, valid_iou = 0.0, 0.0
        with torch.no_grad():
            for imgs, masks in valid_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                logits = model(imgs)
                loss = criterion(logits, masks)
                valid_loss += loss.item() * imgs.size(0)
                valid_iou += hitung_iou(logits, masks) * imgs.size(0)
        valid_loss /= len(valid_ds)
        valid_iou /= len(valid_ds)

        scheduler.step(valid_iou)
        print(f"[unet] Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | "
              f"Valid Loss: {valid_loss:.4f} | Valid IoU: {valid_iou:.4f}")

        if valid_iou > best_iou:
            best_iou = valid_iou
            torch.save(model.state_dict(), "checkpoints/best_unet_segmentasi.pth")
            print(f"🌟 U-Net terbaik disimpan! IoU: {best_iou:.4f}")

    print(f"\nSelesai. IoU validasi terbaik: {best_iou:.4f}")
    print("(IoU di atas ~0.80 umumnya sudah cukup baik untuk keperluan crop/mask kasar)")


if __name__ == "__main__":
    main()
