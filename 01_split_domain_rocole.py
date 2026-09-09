"""
01_split_domain_rocole.py
============================
Membangun dua "domain" dari RoCoLe untuk menguji generalization gap:

  - Domain LAB  (dipakai untuk TRAIN & VALID): daun di-crop ketat mengikuti
    polygon anotasi, latar belakang di luar daun dijadikan putih polos --
    mensimulasikan kondisi "bersih ala laboratorium".
  - Domain FIELD (dipakai untuk TEST, 100% asing bagi model): foto ASLI
    utuh dengan latar belakang alami kebun -- tidak diproses sama sekali.

PENTING soal kebocoran data: foto dikelompokkan dulu berdasarkan kode
pohon/kebun (pola "C<n>P<n>" di nama file), lalu SELURUH kelompok itu
dialokasikan ke satu domain saja. Jadi tidak ada foto dari pohon yang sama
yang muncul di versi lab (train) maupun versi field (test) -- mencegah
model "menghafal" pohon tertentu alih-alih benar-benar diuji generalisasi
ke kondisi visual yang baru.

Sumber data yang dipakai: RoCoLE-csv.csv, kolom "External ID" (nama file
lokal) dan "Label" (JSON berisi state + polygon per daun).

Cara pakai:
    python 01_split_domain_rocole.py \
        --csv "./Dataset/RoCoLe A robusta coffee leaf images dataset/Annotations/RoCoLE-csv.csv" \
        --photos_dir "./Dataset/RoCoLe A robusta coffee leaf images dataset/Photos" \
        --tujuan ./DATASET/rocole_domain \
        --seed 42
"""

import os
import re
import csv
import json
import random
import argparse
import shutil
from collections import defaultdict
from PIL import Image, ImageDraw

csv.field_size_limit(10_000_000)  # kolom Label bisa panjang


def normalisasi_label(state):
    """RoCoLe pakai dua nilai persis: 'healthy' dan 'unhealthy'.
    PENTING: harus exact match, BUKAN substring check -- karena kata
    'unhealthy' itu sendiri mengandung substring 'healthy' di dalamnya,
    jadi cek `"healthy" in state` akan salah mengklasifikasikan semuanya
    sebagai sehat."""
    state = state.lower().strip()
    if state == "healthy":
        return "sehat"
    return "sakit"  # mencakup 'unhealthy' dan variasi lain apapun


def ambil_kode_pohon(filename):
    """'C10P10E1.jpg' -> 'C10P10' -- dipakai sebagai unit split supaya
    tidak ada kebocoran foto dari pohon yang sama ke domain berbeda."""
    m = re.match(r"(C\d+P\d+)", filename)
    return m.group(1) if m else filename  # fallback: per-file kalau pola tidak cocok


def baca_anotasi(csv_path):
    """Kembalikan list of dict: {filename, state, polygon (list of (x,y))}."""
    hasil = []
    dilewati = 0
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row.get("External ID", "").strip()
            label_raw = row.get("Label", "")
            if not filename or not label_raw or label_raw.strip() in ("", "Skip"):
                dilewati += 1
                continue
            try:
                label = json.loads(label_raw)
            except json.JSONDecodeError:
                dilewati += 1
                continue

            daun_list = label.get("Leaf", [])
            if not daun_list:
                dilewati += 1
                continue

            for daun in daun_list:
                state = daun.get("state", "")
                geometry = daun.get("geometry", [])
                if not state or not geometry:
                    continue
                polygon = [(pt["x"], pt["y"]) for pt in geometry]
                hasil.append({"filename": filename, "state": state, "polygon": polygon})

    print(f"Total entri anotasi terbaca: {len(hasil)} (dilewati: {dilewati})")
    return hasil


def buat_versi_lab(img: Image.Image, polygon):
    """Crop ketat mengikuti bounding box polygon, latar luar polygon
    dijadikan putih polos -- versi 'bersih ala lab'."""
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    box = (max(0, min(xs) - 10), max(0, min(ys) - 10),
           min(img.width, max(xs) + 10), min(img.height, max(ys) + 10))

    mask = Image.new("L", img.size, 0)
    ImageDraw.Draw(mask).polygon(polygon, fill=255)

    putih = Image.new("RGB", img.size, (255, 255, 255))
    hasil = Image.composite(img, putih, mask)
    return hasil.crop(box)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--photos_dir", required=True)
    ap.add_argument("--tujuan", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rasio_train", type=float, default=0.70)
    ap.add_argument("--rasio_valid", type=float, default=0.15)
    # sisanya (default 0.15) -> domain field/test
    args = ap.parse_args()

    anotasi = baca_anotasi(args.csv)

    # Kelompokkan entri anotasi per kode pohon
    per_pohon = defaultdict(list)
    for a in anotasi:
        kode = ambil_kode_pohon(a["filename"])
        per_pohon[kode].append(a)

    kode_list = list(per_pohon.keys())
    random.Random(args.seed).shuffle(kode_list)
    n = len(kode_list)
    b1 = int(n * args.rasio_train)
    b2 = int(n * (args.rasio_train + args.rasio_valid))
    kode_train = set(kode_list[:b1])
    kode_valid = set(kode_list[b1:b2])
    kode_test = set(kode_list[b2:])

    print(f"Jumlah kelompok pohon -> train: {len(kode_train)}, "
          f"valid: {len(kode_valid)}, test(field/asing): {len(kode_test)}")

    counts = defaultdict(int)
    hilang = 0

    for kode, entri_list in per_pohon.items():
        if kode in kode_train:
            split, domain = "train", "lab"
        elif kode in kode_valid:
            split, domain = "valid", "lab"
        else:
            split, domain = "test", "field"

        for entri in entri_list:
            src_path = os.path.join(args.photos_dir, entri["filename"])
            if not os.path.exists(src_path):
                hilang += 1
                continue

            label = normalisasi_label(entri["state"])
            tujuan_dir = os.path.join(args.tujuan, split, label)
            os.makedirs(tujuan_dir, exist_ok=True)

            try:
                img = Image.open(src_path).convert("RGB")
            except Exception as e:
                print(f"  ⚠️ Gagal buka {src_path}: {e}")
                continue

            if domain == "lab":
                out_img = buat_versi_lab(img, entri["polygon"])
            else:
                out_img = img  # field/test: foto asli utuh, tidak diproses

            nama_keluaran = f"{domain}_{entri['filename']}"
            out_img.save(os.path.join(tujuan_dir, nama_keluaran), quality=95)
            counts[(split, label)] += 1

    print("\n=== Ringkasan hasil ===")
    for (split, label), n in sorted(counts.items()):
        print(f"  {split}/{label}: {n} gambar")
    if hilang:
        print(f"\n⚠️ {hilang} entri anotasi tidak ketemu file fotonya di {args.photos_dir}")

    print(f"\n🎉 Selesai. Dataset domain-split ada di: {args.tujuan}")
    print("   train/valid = domain LAB (leaf ter-crop, background putih)")
    print("   test        = domain FIELD (foto asli utuh, ASING bagi model)")
    print("\n   Lanjut jalankan:")
    print(f"   python 02_train_mobilevit.py --data_dir {args.tujuan} --tag mobilevit_labdomain --robust_aug")


if __name__ == "__main__":
    main()
