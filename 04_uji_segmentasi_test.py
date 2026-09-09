"""
04_uji_segmentasi_test.py
============================
Uji hipotesis: apakah penurunan akurasi di domain FIELD disebabkan oleh
background yang rumit? Caranya: ambil foto-foto yang sama persis dengan
test set (field), tapi kali ini di-crop+mask jadi background putih
(pakai anotasi polygon yang sama, gaya yang sama seperti domain LAB saat
training) -- lalu evaluasi model yang SAMA (tidak dilatih ulang) ke versi
tersegmentasi ini.

Kalau akurasi balik naik mendekati angka domain LAB, itu bukti kuat bahwa
background memang biang kerok utama gap generalisasi -- bukan karena
kualitas foto lain (fokus, sudut, dsb).

Cara pakai:
    python 04_uji_segmentasi_test.py \
        --csv "./Dataset/RoCoLe A robusta coffee leaf images dataset/Annotations/RoCoLE-csv.csv" \
        --photos_dir "./Dataset/RoCoLe A robusta coffee leaf images dataset/Photos" \
        --data_dir ./DATASET/rocole_domain \
        --tag mobilevit_labdomain \
        --seed 42
"""

import os
import argparse
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from importlib import import_module

split_mod = import_module("01_split_domain_rocole")
train_mod = import_module("02_train_mobilevit")


def buat_test_segmented(csv_path, photos_dir, tujuan_root, seed, rasio_train, rasio_valid):
    """Reproduksi PERSIS split kelompok pohon dari 01_split_domain_rocole.py
    (seed & rasio sama), lalu KHUSUS untuk kelompok yang jatuh ke test,
    buat versi ter-crop+mask (bukan foto asli utuh seperti sebelumnya)."""
    import random
    from collections import defaultdict

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

    print(f"Jumlah kelompok pohon domain test (field): {len(kode_test)}")

    tujuan_dir_base = os.path.join(tujuan_root, "test_segmented")
    counts = {}
    from PIL import Image

    for kode in kode_test:
        for entri in per_pohon[kode]:
            src_path = os.path.join(photos_dir, entri["filename"])
            if not os.path.exists(src_path):
                continue
            label = split_mod.normalisasi_label(entri["state"])
            tujuan_dir = os.path.join(tujuan_dir_base, label)
            os.makedirs(tujuan_dir, exist_ok=True)

            img = Image.open(src_path).convert("RGB")
            out_img = split_mod.buat_versi_lab(img, entri["polygon"])
            out_img.save(os.path.join(tujuan_dir, f"seg_{entri['filename']}"), quality=95)
            counts[label] = counts.get(label, 0) + 1

    print("Hasil test_segmented:", counts)
    return tujuan_dir_base


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
    ap.add_argument("--csv", required=True)
    ap.add_argument("--photos_dir", required=True)
    ap.add_argument("--data_dir", required=True, help="folder yang sama dengan hasil 01_split_domain_rocole.py")
    ap.add_argument("--tag", default="mobilevit_labdomain")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rasio_train", type=float, default=0.70)
    ap.add_argument("--rasio_valid", type=float, default=0.15)
    args = ap.parse_args()

    print("=== TAHAP 1: Membuat test_segmented (crop+mask, background putih) ===")
    test_seg_dir = buat_test_segmented(
        args.csv, args.photos_dir, args.data_dir, args.seed, args.rasio_train, args.rasio_valid
    )

    print("\n=== TAHAP 2: Evaluasi model ke test_segmented ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eval_tf = transforms.Compose([
        transforms.Resize((train_mod.IMG_SIZE, train_mod.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    model = train_mod.DetectorModel(num_classes=2).to(device)
    model.load_state_dict(torch.load(f"checkpoints/best_{args.tag}_model.pth", map_location=device))

    ds = datasets.ImageFolder(test_seg_dir, eval_tf)
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=2)
    y_true, y_pred = evaluate(model, loader, device)
    acc = accuracy_score(y_true, y_pred)

    print(f"\nJumlah gambar: {len(ds)} | Kelas: {ds.classes}")
    print(f"Akurasi TEST_SEGMENTED (foto sama, background dihapus): {acc*100:.2f}%")
    print(classification_report(y_true, y_pred, target_names=ds.classes))
    print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))

    print("\n" + "=" * 70)
    print("BANDINGKAN TIGA ANGKA INI:")
    print("  VALID (lab)          : 96.55%  <- dari training kemarin")
    print("  TEST (field, utuh)   : 72.88%  <- dari evaluasi kemarin")
    print(f"  TEST_SEGMENTED (baru): {acc*100:.2f}%  <- foto SAMA dengan TEST, cuma background dihapus")
    print("=" * 70)
    print("\nKalau TEST_SEGMENTED mendekati VALID -> background TERBUKTI jadi biang")
    print("kerok utama, dan segmentasi otomatis sebelum klasifikasi adalah solusi")
    print("praktis yang bisa direkomendasikan untuk deployment.")
    print("Kalau TEST_SEGMENTED masih rendah -> ada faktor lain juga yang berperan")
    print("(mis. sudut foto, pencahayaan, kualitas kamera), bukan cuma background.")


if __name__ == "__main__":
    main()
