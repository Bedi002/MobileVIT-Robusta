"""
02_train_mobilevit.py
========================
Training MobileViT untuk klasifikasi penyakit daun kopi (RoCoLe), dengan
metodologi yang sama persis dengan yang sudah terbukti di skripsi utama:
hyperparameter identik antar-eksperimen, augmentasi kompresi JPEG (robust_aug),
dan dukungan logging DagsHub/MLflow.

Asumsi struktur data (ImageFolder standar, sama seperti sebelumnya):
    <data_dir>/train/sehat/...      <data_dir>/train/rust/...
    <data_dir>/valid/sehat/...      <data_dir>/valid/rust/...
    <data_dir>/test/sehat/...       <data_dir>/test/rust/...

Cara pakai:
    python 02_train_mobilevit.py --data_dir ./DATASET/rocole_lab --robust_aug
    python 02_train_mobilevit.py --data_dir ./DATASET/rocole_field --robust_aug

(Folder data_dir yang mana persisnya -- "lab" vs "field" -- baru bisa
dipastikan setelah kita tahu format anotasi RoCoLe dari hasil inspeksi.)

Install dependency (kalau timm versi lama belum punya mobilevit):
    pip install --upgrade timm
"""

import os
import io
import time
import random
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import timm
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ----------------------------------------------------------------------
# HYPERPARAMETER TETAP -- samakan persis antar semua eksperimen
# ----------------------------------------------------------------------
IMG_SIZE = 256          # MobileViT umumnya dilatih di 256x256 (bukan 224 seperti MobileNetV4)
BATCH_SIZE = 16          # lebih kecil dari eksperimen sebelumnya -- MobileViT lebih berat
                         # dari MobileNetV4-Small, dan RoCoLe cuma ~1500 gambar (dataset kecil)
EPOCHS = 30              # dataset kecil -> butuh epoch lebih banyak, tapi tetap cepat karena
                         # dataset kecil juga (1 epoch = beberapa detik-menit saja)
LR = 3e-5                # MobileViT pretrained biasanya perlu LR lebih kecil dari MobileNetV4
WEIGHT_DECAY = 1e-2
LABEL_SMOOTHING = 0.1
BACKBONE_NAME = "mobilevit_s"   # varian small; kalau VRAM 6GB kurang, ganti ke "mobilevit_xs"
SEED = 42


class RandomJPEGCompression:
    """Sama seperti di skripsi utama: paksa model terbiasa artefak kompresi."""

    def __init__(self, quality_range=(30, 95), p=0.5):
        self.quality_range = quality_range
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        quality = random.randint(*self.quality_range)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        return Image.open(buf).convert("RGB")


class DetectorModel(nn.Module):
    """Wrapper backbone (MobileViT ATAU arsitektur lain) + classifier head.
    Backbone dibuat configurable supaya baseline (mis. MobileNetV2) bisa
    dilatih dengan pipeline & hyperparameter yang IDENTIK -- perbandingan adil,
    persis prinsip yang dipakai di skripsi utama (baseline vs CBAM)."""

    def __init__(self, num_classes=2, backbone_name=None, dropout=0.3):
        super().__init__()
        backbone_name = backbone_name or BACKBONE_NAME
        self.backbone_name = backbone_name
        self.backbone = timm.create_model(
            backbone_name, pretrained=True, num_classes=0, global_pool="avg"
        )
        with torch.no_grad():
            dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
            feature_dim = self.backbone(dummy).shape[1]
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.dropout(x)
        return self.fc(x)


def build_dataloaders(data_dir, robust_aug=False):
    train_transforms_list = [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=20),          # daun bisa difoto dari sudut apapun
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),  # variasi cahaya lapangan
        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
    ]
    if robust_aug:
        train_transforms_list.append(RandomJPEGCompression(quality_range=(30, 95), p=0.5))
    train_transforms_list += [
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
    train_tf = transforms.Compose(train_transforms_list)
    eval_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_ds = datasets.ImageFolder(os.path.join(data_dir, "train"), train_tf)
    valid_ds = datasets.ImageFolder(os.path.join(data_dir, "valid"), eval_tf)
    test_ds = datasets.ImageFolder(os.path.join(data_dir, "test"), eval_tf)

    loaders = {
        "train": DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2),
        "valid": DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2),
        "test": DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2),
    }
    sizes = {"train": len(train_ds), "valid": len(valid_ds), "test": len(test_ds)}
    return loaders, sizes, train_ds.classes


def train(model, loaders, sizes, device, tag, backbone_name, use_mlflow=False, mlflow_run_name=None):
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    os.makedirs("checkpoints", exist_ok=True)
    best_acc = 0.0
    since = time.time()

    mlflow_ctx = None
    if use_mlflow:
        import mlflow
        mlflow_ctx = mlflow.start_run(run_name=mlflow_run_name or f"Training_{tag}")
        mlflow_ctx.__enter__()
        mlflow.log_param("model_tag", tag)
        mlflow.log_param("backbone", backbone_name)
        mlflow.log_param("learning_rate", LR)
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("epochs", EPOCHS)
        mlflow.log_param("img_size", IMG_SIZE)
        mlflow.log_param("seed", SEED)

    for epoch in range(EPOCHS):
        print(f"\n[{tag}] Epoch {epoch + 1}/{EPOCHS}")
        print("-" * 10)
        for phase in ["train", "valid"]:
            model.train() if phase == "train" else model.eval()
            running_loss, running_corrects = 0.0, 0

            for inputs, labels in loaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / sizes[phase]
            epoch_acc = running_corrects.double() / sizes[phase]
            print(f"{phase.upper()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            if use_mlflow:
                import mlflow
                mlflow.log_metric(f"{phase}_loss", epoch_loss, step=epoch)
                mlflow.log_metric(f"{phase}_accuracy", float(epoch_acc), step=epoch)

            if phase == "valid" and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), f"checkpoints/best_{tag}_model.pth")
                print(f"🌟 Model terbaik ({tag}) disimpan! Acc: {best_acc:.4f}")

        scheduler.step()

    if use_mlflow:
        import mlflow
        mlflow.log_metric("best_valid_accuracy", float(best_acc))
        mlflow_ctx.__exit__(None, None, None)

    elapsed = time.time() - since
    print(f"\n[{tag}] Training selesai dalam {elapsed // 60:.0f}m {elapsed % 60:.0f}s")
    print(f"[{tag}] Akurasi validasi terbaik: {best_acc:.4f}")
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--backbone", default="mobilevit_s",
                     help="Nama model timm. Default: mobilevit_s. Untuk baseline pembanding, "
                          "pakai 'mobilenetv2_100' -- hyperparameter lain TETAP SAMA demi keadilan.")
    ap.add_argument("--tag", default="mobilevit",
                     help="Nama tag untuk checkpoint & MLflow run, mis. 'mobilevit_lab' atau 'mobilevit_field'")
    ap.add_argument("--dagshub_repo_owner", default="Bedi002",
                     help="Default: Bedi002 (ubah kalau perlu ganti akun)")
    ap.add_argument("--dagshub_repo_name", default="MobileVIT-Robusta",
                     help="Default: MobileVIT-Robusta (ubah kalau perlu ganti repo)")
    ap.add_argument("--no_dagshub", action="store_true",
                     help="Matikan koneksi DagsHub sama sekali untuk run ini (mis. buat tes cepat offline)")
    ap.add_argument("--robust_aug", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(SEED)
    random.seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Perangkat aktif: {device}")
    print(f"Backbone: {args.backbone}")

    use_mlflow = False
    if not args.no_dagshub and args.dagshub_repo_owner and args.dagshub_repo_name:
        import dagshub
        import mlflow
        dagshub.init(repo_owner=args.dagshub_repo_owner,
                      repo_name=args.dagshub_repo_name, mlflow=True)
        mlflow.set_tracking_uri(
            f"https://dagshub.com/{args.dagshub_repo_owner}/{args.dagshub_repo_name}.mlflow"
        )
        use_mlflow = True
        print(f"📡 Terhubung ke DagsHub: {args.dagshub_repo_owner}/{args.dagshub_repo_name}")

    loaders, sizes, classes = build_dataloaders(args.data_dir, robust_aug=args.robust_aug)
    print(f"Kelas: {classes} | Train: {sizes['train']} | Valid: {sizes['valid']} | Test: {sizes['test']}")

    model = DetectorModel(num_classes=len(classes), backbone_name=args.backbone).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Jumlah parameter: {n_params:,}")

    train(model, loaders, sizes, device, tag=args.tag, backbone_name=args.backbone, use_mlflow=use_mlflow,
          mlflow_run_name=f"Training_{args.tag}")


if __name__ == "__main__":
    main()
