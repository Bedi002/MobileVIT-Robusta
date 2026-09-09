"""
07_bandingkan_arsitektur.py
==============================
Membandingkan MobileViT (usulan) vs MobileNetV2 (baseline pembanding) --
DUA-DUANYA dilatih dan dievaluasi pada split data yang PERSIS SAMA (domain
lab untuk train/valid, domain field untuk test). Ini penting supaya
perbandingan benar-benar adil: bukan membandingkan ke angka dari paper
orang lain yang datasetnya/split-nya beda, tapi head-to-head di kondisi
yang identik.

Prasyarat: kedua model harus sudah dilatih lebih dulu dengan:
    python 02_train_mobilevit.py --data_dir ./DATASET/rocole_domain --backbone mobilevit_s --tag mobilevit_labdomain --robust_aug
    python 02_train_mobilevit.py --data_dir ./DATASET/rocole_domain --backbone mobilenetv2_100 --tag mobilenetv2_baseline --robust_aug

Cara pakai:
    python 07_bandingkan_arsitektur.py --data_dir ./DATASET/rocole_domain
"""

import argparse
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from importlib import import_module

train_mod = import_module("02_train_mobilevit")


def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.numpy())
            y_pred.extend(preds.cpu().numpy())
    return y_true, y_pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--tag_mobilevit", default="mobilevit_labdomain")
    ap.add_argument("--tag_mobilenetv2", default="mobilenetv2_baseline")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eval_tf = transforms.Compose([
        transforms.Resize((train_mod.IMG_SIZE, train_mod.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    model_configs = [
        ("MobileViT (usulan)", "mobilevit_s", args.tag_mobilevit),
        ("MobileNetV2 (baseline)", "mobilenetv2_100", args.tag_mobilenetv2),
    ]

    hasil = {}
    for label, backbone, tag in model_configs:
        print(f"\n>> Memuat {label} (backbone={backbone}, tag={tag}) ...")
        model = train_mod.DetectorModel(num_classes=2, backbone_name=backbone).to(device)
        model.load_state_dict(torch.load(f"checkpoints/best_{tag}_model.pth", map_location=device))
        n_params = sum(p.numel() for p in model.parameters())

        hasil[label] = {"n_params": n_params}
        for split in ["valid", "test"]:
            ds = datasets.ImageFolder(f"{args.data_dir}/{split}", eval_tf)
            loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=2)
            y_true, y_pred = evaluate(model, loader, device)
            hasil[label][f"{split}_acc"] = accuracy_score(y_true, y_pred)
            hasil[label][f"{split}_f1"] = f1_score(y_true, y_pred, average="weighted")

    print("\n" + "=" * 78)
    print(f"{'Metrik':<28}{'MobileViT (usulan)':<22}{'MobileNetV2 (baseline)':<22}")
    print("-" * 78)
    print(f"{'Jumlah parameter':<28}{hasil['MobileViT (usulan)']['n_params']:<22,}{hasil['MobileNetV2 (baseline)']['n_params']:<22,}")
    print(f"{'Akurasi VALID (lab)':<28}{hasil['MobileViT (usulan)']['valid_acc']*100:<22.2f}{hasil['MobileNetV2 (baseline)']['valid_acc']*100:<22.2f}")
    print(f"{'Akurasi TEST (field)':<28}{hasil['MobileViT (usulan)']['test_acc']*100:<22.2f}{hasil['MobileNetV2 (baseline)']['test_acc']*100:<22.2f}")
    gap_vit = (hasil['MobileViT (usulan)']['valid_acc'] - hasil['MobileViT (usulan)']['test_acc']) * 100
    gap_v2 = (hasil['MobileNetV2 (baseline)']['valid_acc'] - hasil['MobileNetV2 (baseline)']['test_acc']) * 100
    print(f"{'Generalization gap (poin %)':<28}{gap_vit:<22.2f}{gap_v2:<22.2f}")
    print("=" * 78)

    print("\nCatatan interpretasi:")
    print("- Kalau akurasi TEST MobileViT lebih tinggi DAN gap-nya lebih kecil dari")
    print("  MobileNetV2, itu bukti MobileViT punya keunggulan generalisasi nyata,")
    print("  bukan cuma ikut-ikutan tren arsitektur.")
    print("- Kalau keduanya mirip, itu tetap temuan valid: generalization gap di")
    print("  kasus ini lebih dipengaruhi oleh domain (background) daripada pilihan")
    print("  arsitektur -- justru memperkuat argumen bahwa solusi pipeline")
    print("  segmentasi (bukan sekadar ganti arsitektur) yang benar-benar diperlukan.")


if __name__ == "__main__":
    main()
