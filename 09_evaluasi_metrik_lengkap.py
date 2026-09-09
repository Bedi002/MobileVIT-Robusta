"""
09_evaluasi_metrik_lengkap.py
================================
Menjawab poin penyempurnaan #1: hitung Precision/Recall/F1 PER KELAS
(bukan cuma akurasi) untuk MobileViT dan MobileNetV2, di KEEMPAT skenario
yang sudah pernah diuji:
    1. VALID (lab)
    2. TEST field mentah (tanpa proses)
    3. TEST_SEGMENTED (anotasi manual -- batas atas ideal)
    4. PIPELINE deployment (auto-segment U-Net, skenario realistis)

TIDAK PERLU TRAINING ULANG -- script ini hanya memuat checkpoint yang
sudah ada dan menjalankan evaluasi. Skenario 3 butuh folder test_segmented
(dari 04_uji_segmentasi_test.py); kalau belum ada, script ini akan
membuatnya otomatis lebih dulu (reproduksi split yang sama, deterministik).

Hasil disimpan sebagai:
    - metrik_lengkap.csv   (semua angka precision/recall/F1/accuracy per baris)
    - dicetak juga sebagai tabel ringkas di terminal

Cara pakai:
    python 09_evaluasi_metrik_lengkap.py \
        --csv "./Dataset/RoCoLe A robusta coffee leaf images dataset/Annotations/RoCoLE-csv.csv" \
        --photos_dir "./Dataset/RoCoLe A robusta coffee leaf images dataset/Photos" \
        --data_dir ./DATASET/rocole_domain \
        --unet_checkpoint checkpoints/best_unet_segmentasi.pth \
        --tag_mobilevit mobilevit_labdomain \
        --tag_mobilenetv2 mobilenetv2_baseline \
        --seed 42
"""

import os
import csv
import argparse
import torch
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, accuracy_score
from importlib import import_module

split_mod = import_module("01_split_domain_rocole")
train_mod = import_module("02_train_mobilevit")
seg_mod = import_module("04_uji_segmentasi_test")
pipeline_mod = import_module("06_pipeline_deployment_real")
unet_mod = import_module("05_train_unet_segmentasi")

KELAS = ["sakit", "sehat"]  # urutan ImageFolder alfabetis


def evaluasi_imagefolder(model, folder_path, tf, device):
    ds = datasets.ImageFolder(folder_path, tf)
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=2)
    y_true, y_pred = [], []
    model.eval()
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            y_true.extend(ds.classes[i] for i in labels.numpy())
            y_pred.extend(ds.classes[i] for i in preds.cpu().numpy())
    return y_true, y_pred


def evaluasi_pipeline(model, unet, entri_test, photos_dir, classify_tf, device):
    y_true, y_pred = [], []
    model.eval()
    for entri in entri_test:
        img_path = os.path.join(photos_dir, entri["filename"])
        if not os.path.exists(img_path):
            continue
        img = Image.open(img_path).convert("RGB")
        label_asli = split_mod.normalisasi_label(entri["state"])
        hasil_crop, _ = pipeline_mod.auto_segment_dan_crop(unet, img, device)
        x = classify_tf(hasil_crop.convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(x)
            pred_idx = torch.argmax(logits, dim=1).item()
        y_true.append(label_asli)
        y_pred.append(KELAS[pred_idx])
    return y_true, y_pred


def cetak_dan_simpan(rows_writer, label_arsitektur, nama_skenario, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, labels=KELAS, target_names=KELAS,
                                    output_dict=True, zero_division=0)
    print(f"\n--- {label_arsitektur} | {nama_skenario} ---")
    print(f"Akurasi: {acc*100:.2f}% | n={len(y_true)}")
    print(classification_report(y_true, y_pred, labels=KELAS, target_names=KELAS, zero_division=0))

    for kelas in KELAS:
        rows_writer.writerow({
            "arsitektur": label_arsitektur,
            "skenario": nama_skenario,
            "kelas": kelas,
            "precision": f"{report[kelas]['precision']:.4f}",
            "recall": f"{report[kelas]['recall']:.4f}",
            "f1_score": f"{report[kelas]['f1-score']:.4f}",
            "support": report[kelas]["support"],
            "akurasi_keseluruhan": f"{acc:.4f}",
        })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--photos_dir", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--unet_checkpoint", default="checkpoints/best_unet_segmentasi.pth")
    ap.add_argument("--tag_mobilevit", default="mobilevit_labdomain")
    ap.add_argument("--tag_mobilenetv2", default="mobilenetv2_baseline")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rasio_train", type=float, default=0.70)
    ap.add_argument("--rasio_valid", type=float, default=0.15)
    ap.add_argument("--output_csv", default="metrik_lengkap.csv")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Perangkat aktif: {device}")

    eval_tf = transforms.Compose([
        transforms.Resize((train_mod.IMG_SIZE, train_mod.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # Siapkan folder TEST_SEGMENTED (buat otomatis kalau belum ada)
    test_seg_dir = os.path.join(args.data_dir, "test_segmented")
    if not os.path.isdir(test_seg_dir):
        print("\nFolder test_segmented belum ada -- membuatnya sekarang (reproduksi split deterministik)...")
        test_seg_dir = seg_mod.buat_test_segmented(
            args.csv, args.photos_dir, args.data_dir, args.seed, args.rasio_train, args.rasio_valid
        )
    else:
        print(f"\nFolder test_segmented sudah ada: {test_seg_dir} (dipakai langsung)")

    # Siapkan U-Net + kelompok test untuk skenario PIPELINE
    print("\nMemuat U-Net segmentasi...")
    unet = unet_mod.UNetRingan().to(device)
    unet.load_state_dict(torch.load(args.unet_checkpoint, map_location=device))
    unet.eval()
    entri_test = pipeline_mod.ambil_kelompok_test(args.csv, args.seed, args.rasio_train, args.rasio_valid)
    print(f"Jumlah foto field untuk skenario PIPELINE: {len(entri_test)}")

    model_configs = [
        ("MobileViT", "mobilevit_s", args.tag_mobilevit),
        ("MobileNetV2", "mobilenetv2_100", args.tag_mobilenetv2),
    ]

    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["arsitektur", "skenario", "kelas", "precision", "recall",
                      "f1_score", "support", "akurasi_keseluruhan"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for label, backbone, tag in model_configs:
            print(f"\n{'='*70}\nMemuat {label} (backbone={backbone}, tag={tag})\n{'='*70}")
            model = train_mod.DetectorModel(num_classes=2, backbone_name=backbone).to(device)
            model.load_state_dict(torch.load(f"checkpoints/best_{tag}_model.pth", map_location=device))

            y_true, y_pred = evaluasi_imagefolder(model, f"{args.data_dir}/valid", eval_tf, device)
            cetak_dan_simpan(writer, label, "VALID (lab)", y_true, y_pred)

            y_true, y_pred = evaluasi_imagefolder(model, f"{args.data_dir}/test", eval_tf, device)
            cetak_dan_simpan(writer, label, "TEST field mentah", y_true, y_pred)

            y_true, y_pred = evaluasi_imagefolder(model, test_seg_dir, eval_tf, device)
            cetak_dan_simpan(writer, label, "TEST_SEGMENTED (anotasi manual)", y_true, y_pred)

            y_true, y_pred = evaluasi_pipeline(model, unet, entri_test, args.photos_dir, eval_tf, device)
            cetak_dan_simpan(writer, label, "PIPELINE deployment (auto-segment)", y_true, y_pred)

    print(f"\n🎉 Selesai. Semua angka precision/recall/F1 per kelas tersimpan di: {args.output_csv}")
    print("   Tinggal dijadikan tabel di Bab IV -- sudah lengkap, tidak perlu training ulang.")


if __name__ == "__main__":
    main()
