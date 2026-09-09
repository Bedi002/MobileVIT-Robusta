"""
08_bandingkan_pipeline_deployment.py
=======================================
Menjawab pertanyaan: apakah solusi pipeline segmentasi otomatis (U-Net)
menolong SEMUA arsitektur classifier, atau cuma kebetulan cocok untuk
MobileViT saja? Script ini menjalankan pipeline auto-segment + klasifikasi
LENGKAP untuk MobileViT maupun MobileNetV2 (dua-duanya pakai U-Net yang
SAMA, satu-satunya yang beda cuma classifier di ujung pipeline), lalu
menyandingkan hasilnya dalam satu tabel bersama angka-angka sebelumnya.

Prasyarat: sudah menjalankan sebelumnya --
    python 05_train_unet_segmentasi.py ...
    python 02_train_mobilevit.py --backbone mobilevit_s --tag mobilevit_labdomain ...
    python 02_train_mobilevit.py --backbone mobilenetv2_100 --tag mobilenetv2_baseline ...

Cara pakai:
    python 08_bandingkan_pipeline_deployment.py \
        --csv "./Dataset/RoCoLe A robusta coffee leaf images dataset/Annotations/RoCoLE-csv.csv" \
        --photos_dir "./Dataset/RoCoLe A robusta coffee leaf images dataset/Photos" \
        --seed 42
"""

import argparse
import torch
from PIL import Image
from torchvision import transforms
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from importlib import import_module

split_mod = import_module("01_split_domain_rocole")
unet_mod = import_module("05_train_unet_segmentasi")
train_mod = import_module("02_train_mobilevit")
pipeline_mod = import_module("06_pipeline_deployment_real")


def jalankan_satu_model(label, backbone, tag, unet, entri_test, photos_dir, device):
    print(f"\n>> Menjalankan pipeline penuh untuk {label} (backbone={backbone}) ...")
    classifier = train_mod.DetectorModel(num_classes=2, backbone_name=backbone).to(device)
    classifier.load_state_dict(torch.load(f"checkpoints/best_{tag}_model.pth", map_location=device))
    classifier.eval()

    classify_tf = transforms.Compose([
        transforms.Resize((train_mod.IMG_SIZE, train_mod.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    y_true, y_pred = [], []
    for entri in entri_test:
        import os
        img_path = os.path.join(photos_dir, entri["filename"])
        if not os.path.exists(img_path):
            continue
        img = Image.open(img_path).convert("RGB")
        label_asli = split_mod.normalisasi_label(entri["state"])

        hasil_crop, _ = pipeline_mod.auto_segment_dan_crop(unet, img, device)

        x = classify_tf(hasil_crop.convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = classifier(x)
            pred_idx = torch.argmax(logits, dim=1).item()
        label_pred = ["sakit", "sehat"][pred_idx]

        y_true.append(label_asli)
        y_pred.append(label_pred)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")
    return acc, f1, confusion_matrix(y_true, y_pred, labels=["sakit", "sehat"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--photos_dir", required=True)
    ap.add_argument("--unet_checkpoint", default="checkpoints/best_unet_segmentasi.pth")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rasio_train", type=float, default=0.70)
    ap.add_argument("--rasio_valid", type=float, default=0.15)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Perangkat aktif: {device}")

    unet = unet_mod.UNetRingan().to(device)
    unet.load_state_dict(torch.load(args.unet_checkpoint, map_location=device))
    unet.eval()

    entri_test = pipeline_mod.ambil_kelompok_test(args.csv, args.seed, args.rasio_train, args.rasio_valid)
    print(f"Jumlah foto field (test, ASING bagi semua model): {len(entri_test)}")

    model_configs = [
        ("MobileViT", "mobilevit_s", "mobilevit_labdomain"),
        ("MobileNetV2", "mobilenetv2_100", "mobilenetv2_baseline"),
    ]

    hasil = {}
    for label, backbone, tag in model_configs:
        acc, f1, cm = jalankan_satu_model(label, backbone, tag, unet, entri_test, args.photos_dir, device)
        hasil[label] = {"acc": acc, "f1": f1, "cm": cm}
        print(f"   {label}: akurasi pipeline = {acc*100:.2f}%")
        print(f"   Confusion matrix:\n{cm}")

    print("\n" + "=" * 78)
    print("RINGKASAN LENGKAP -- SEMUA SKENARIO YANG SUDAH DIUJI")
    print("=" * 78)
    print(f"{'Skenario':<45}{'MobileViT':<16}{'MobileNetV2':<16}")
    print("-" * 78)
    print(f"{'VALID (lab)':<45}{'96.55':<16}{'94.40':<16}")
    print(f"{'TEST field mentah (tanpa proses)':<45}{'72.88':<16}{'79.66':<16}")
    print(f"{'PIPELINE DEPLOYMENT (auto-segment)':<45}{hasil['MobileViT']['acc']*100:<16.2f}{hasil['MobileNetV2']['acc']*100:<16.2f}")
    print("=" * 78)

    naik_vit = hasil['MobileViT']['acc']*100 - 72.88
    naik_v2 = hasil['MobileNetV2']['acc']*100 - 79.66
    print(f"\nKenaikan akurasi berkat pipeline segmentasi:")
    print(f"  MobileViT   : {naik_vit:+.2f} poin")
    print(f"  MobileNetV2 : {naik_v2:+.2f} poin")
    print("\nKalau KEDUA model sama-sama naik signifikan, itu bukti pipeline")
    print("segmentasi ini ARSITEKTUR-AGNOSTIK -- solusi yang general, bukan")
    print("kebetulan cocok untuk satu model saja. Ini nilai kontribusi yang lebih")
    print("tinggi untuk dilaporkan di Bab IV/V dibanding solusi yang spesifik")
    print("cuma jalan di satu arsitektur.")


if __name__ == "__main__":
    main()
