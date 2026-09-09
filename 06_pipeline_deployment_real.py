"""
06_pipeline_deployment_real.py
================================
Pipeline paling mendekati kondisi deployment nyata: foto lapangan mentah
(field, belum pernah dilihat model manapun -- baik classifier maupun
segmenter) -> di-segmentasi OTOMATIS pakai U-Net terlatih (BUKAN anotasi
manual) -> di-crop & mask jadi background putih -> diklasifikasi pakai
MobileViT yang sudah dilatih sebelumnya.

Ini beda dari 04_uji_segmentasi_test.py: yang itu masih "curang" karena
pakai anotasi polygon manusia (ground truth) untuk bikin mask. Script ini
pakai prediksi model segmentasi sendiri -- persis skenario dunia nyata
saat aplikasi menerima foto baru dari petani tanpa ada anotasi apapun.

Cara pakai:
    python 06_pipeline_deployment_real.py \
        --csv "./Dataset/RoCoLe A robusta coffee leaf images dataset/Annotations/RoCoLE-csv.csv" \
        --photos_dir "./Dataset/RoCoLe A robusta coffee leaf images dataset/Photos" \
        --unet_checkpoint checkpoints/best_unet_segmentasi.pth \
        --classifier_tag mobilevit_labdomain \
        --seed 42
"""

import os
import random
import argparse
from collections import defaultdict

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from importlib import import_module

split_mod = import_module("01_split_domain_rocole")
unet_mod = import_module("05_train_unet_segmentasi")
train_mod = import_module("02_train_mobilevit")


def ambil_kelompok_test(csv_path, seed, rasio_train, rasio_valid):
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
    kode_test = set(kode_list[b2:])

    return [e for k in kode_test for e in per_pohon[k]]


def auto_segment_dan_crop(unet, img: Image.Image, device, threshold=0.5):
    """Jalankan U-Net untuk memprediksi mask daun, lalu crop & putihkan
    background di luar mask -- SEPENUHNYA otomatis, tidak pakai anotasi."""
    ukuran_asli = img.size  # (width, height)

    img_resized = img.resize((unet_mod.IMG_SIZE, unet_mod.IMG_SIZE))
    arr = np.array(img_resized, dtype=np.float32) / 255.0
    arr = (arr - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    tensor = torch.from_numpy(arr.transpose(2, 0, 1)).float().unsqueeze(0).to(device)

    with torch.no_grad():
        logits = unet(tensor)
        probs = torch.sigmoid(logits)[0, 0].cpu().numpy()

    mask_kecil = (probs > threshold).astype(np.uint8) * 255
    mask_img = Image.fromarray(mask_kecil).resize(ukuran_asli)  # skalakan balik ke ukuran asli
    mask_arr = np.array(mask_img) > 127

    if not mask_arr.any():
        # Gagal deteksi apapun -- kembalikan gambar asli utuh, jangan crash
        return img, False

    ys, xs = np.where(mask_arr)
    box = (max(0, xs.min() - 10), max(0, ys.min() - 10),
           min(ukuran_asli[0], xs.max() + 10), min(ukuran_asli[1], ys.max() + 10))

    img_arr = np.array(img)
    putih = np.full_like(img_arr, 255)
    hasil_arr = np.where(mask_arr[:, :, None], img_arr, putih)
    hasil_img = Image.fromarray(hasil_arr.astype(np.uint8)).crop(box)
    return hasil_img, True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--photos_dir", required=True)
    ap.add_argument("--unet_checkpoint", default="checkpoints/best_unet_segmentasi.pth")
    ap.add_argument("--classifier_tag", default="mobilevit_labdomain")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rasio_train", type=float, default=0.70)
    ap.add_argument("--rasio_valid", type=float, default=0.15)
    ap.add_argument("--simpan_contoh", type=int, default=8,
                     help="Simpan sekian contoh hasil auto-crop untuk inspeksi visual")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Perangkat aktif: {device}")

    print("\n=== Memuat model segmentasi (U-Net) ===")
    unet = unet_mod.UNetRingan().to(device)
    unet.load_state_dict(torch.load(args.unet_checkpoint, map_location=device))
    unet.eval()

    print("=== Memuat model klasifikasi (MobileViT) ===")
    classifier = train_mod.DetectorModel(num_classes=2).to(device)
    classifier.load_state_dict(
        torch.load(f"checkpoints/best_{args.classifier_tag}_model.pth", map_location=device)
    )
    classifier.eval()

    entri_test = ambil_kelompok_test(args.csv, args.seed, args.rasio_train, args.rasio_valid)
    print(f"\nJumlah foto field (test, ASING bagi kedua model): {len(entri_test)}")

    classify_tf = transforms.Compose([
        transforms.Resize((train_mod.IMG_SIZE, train_mod.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    os.makedirs("contoh_auto_segmentasi", exist_ok=True)
    y_true, y_pred = [], []
    gagal_deteksi = 0

    for i, entri in enumerate(entri_test):
        img_path = os.path.join(args.photos_dir, entri["filename"])
        if not os.path.exists(img_path):
            continue
        img = Image.open(img_path).convert("RGB")
        label_asli = split_mod.normalisasi_label(entri["state"])

        hasil_crop, berhasil = auto_segment_dan_crop(unet, img, device)
        if not berhasil:
            gagal_deteksi += 1

        if i < args.simpan_contoh:
            hasil_crop.save(f"contoh_auto_segmentasi/contoh_{i}_{label_asli}.jpg")

        x = classify_tf(hasil_crop.convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = classifier(x)
            pred_idx = torch.argmax(logits, dim=1).item()
        # urutan kelas ImageFolder alfabetis: ['sakit', 'sehat']
        label_pred = ["sakit", "sehat"][pred_idx]

        y_true.append(label_asli)
        y_pred.append(label_pred)

    acc = accuracy_score(y_true, y_pred)
    print(f"\n=== HASIL PIPELINE DEPLOYMENT (auto-segment + klasifikasi) ===")
    print(f"Akurasi: {acc*100:.2f}%")
    print(f"Gagal deteksi daun sama sekali: {gagal_deteksi} foto (pakai foto utuh sbg fallback)")
    print(classification_report(y_true, y_pred, labels=["sakit", "sehat"]))
    print("Confusion matrix:\n", confusion_matrix(y_true, y_pred, labels=["sakit", "sehat"]))

    print("\n" + "=" * 70)
    print("BANDINGKAN EMPAT ANGKA:")
    print("  VALID (lab)                      : 96.55%")
    print("  TEST field mentah (tanpa proses)  : 72.88%")
    print("  TEST_SEGMENTED (anotasi manual)   : 95.76%  <- batas atas ideal")
    print(f"  PIPELINE DEPLOYMENT (auto-segment): {acc*100:.2f}%  <- REALISTIS, ini yang penting")
    print("=" * 70)
    print(f"\nContoh hasil crop otomatis disimpan di folder 'contoh_auto_segmentasi/'")
    print("-- cek visual beberapa biar tahu separah apa kualitas segmentasi otomatisnya.")


if __name__ == "__main__":
    main()
