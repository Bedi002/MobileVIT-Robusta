"""
13_contoh_before_after_segmentasi.py
=======================================
Menghasilkan beberapa contoh pasangan gambar SEBELUM (foto field asli utuh)
dan SESUDAH (hasil crop+mask pakai anotasi polygon manual, gaya yang sama
seperti domain LAB) -- khusus untuk kelompok pohon TEST/FIELD, supaya bisa
dipakai sebagai Gambar 4.2 di skripsi (ilustrasi visual kenapa segmentasi
membantu, sebelum masuk ke pembahasan angka TEST vs TEST_SEGMENTED).

Ini BUKAN evaluasi -- cuma menyimpan gambar untuk inspeksi visual, jadi
tidak perlu checkpoint model apapun.

Reproduksi split: pakai --seed & --rasio_train/--rasio_valid yang SAMA
dengan 01_split_domain_rocole.py & 04_uji_segmentasi_test.py, supaya
kelompok pohon "test/field" yang dipakai persis sama.

Untuk tiap contoh, script ini menyimpan:
  - before_<nama_file>.jpg      (foto field asli, tidak diproses)
  - after_<nama_file>.jpg       (hasil crop+mask pakai anotasi manual)
  - gabungan_<nama_file>.jpg    (before & after digabung sisi-bersisi
                                  dalam SATU gambar -- paling praktis buat
                                  langsung ditempel jadi satu Gambar 4.2)

Cara pakai:
    python 13_contoh_before_after_segmentasi.py \
        --csv "./Dataset/RoCoLe A robusta coffee leaf images dataset/Annotations/RoCoLE-csv.csv" \
        --photos_dir "./Dataset/RoCoLe A robusta coffee leaf images dataset/Photos" \
        --seed 42 \
        --n_contoh 4
"""

import os
import random
import argparse
from collections import defaultdict

from PIL import Image, ImageDraw, ImageFont
from importlib import import_module

split_mod = import_module("01_split_domain_rocole")


def ambil_kelompok_test(csv_path, seed, rasio_train, rasio_valid):
    """Reproduksi PERSIS split kelompok pohon dari 01_split_domain_rocole.py
    (seed & rasio harus sama), kembalikan hanya entri anotasi yang jatuh
    ke kelompok test/field."""
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
    return [e for k in kode_test for e in per_pohon[k]]


def beri_label_gambar(img: Image.Image, teks: str) -> Image.Image:
    """Tempel pita label kecil ('SEBELUM' / 'SESUDAH') di atas gambar,
    supaya kalau nanti gambar dipisah dari gabungan tetap jelas asalnya."""
    img = img.copy()
    draw = ImageDraw.Draw(img)
    tinggi_pita = 28
    draw.rectangle([(0, 0), (img.width, tinggi_pita)], fill=(0, 0, 0))
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    draw.text((8, 6), teks, fill=(255, 255, 255), font=font)
    return img


def gabungkan_sisi_bersisi(img_before: Image.Image, img_after: Image.Image, tinggi_target=400) -> Image.Image:
    """Samakan tinggi kedua gambar, taruh berdampingan dengan sedikit jarak
    putih di tengah -- hasilnya satu file JPG siap tempel sebagai figure."""
    def skala(img):
        rasio = tinggi_target / img.height
        return img.resize((max(1, int(img.width * rasio)), tinggi_target))

    a = beri_label_gambar(skala(img_before), "SEBELUM (field, utuh)")
    b = beri_label_gambar(skala(img_after), "SESUDAH (segmentasi manual)")

    jarak = 12
    gabungan = Image.new("RGB", (a.width + jarak + b.width, tinggi_target), (255, 255, 255))
    gabungan.paste(a, (0, 0))
    gabungan.paste(b, (a.width + jarak, 0))
    return gabungan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--photos_dir", required=True)
    ap.add_argument("--seed", type=int, default=42,
                     help="HARUS sama dengan seed split di 01_split_domain_rocole.py")
    ap.add_argument("--rasio_train", type=float, default=0.70)
    ap.add_argument("--rasio_valid", type=float, default=0.15)
    ap.add_argument("--n_contoh", type=int, default=4,
                     help="Jumlah pasang contoh yang mau disimpan (diusahakan berimbang sehat/sakit)")
    ap.add_argument("--tujuan", default="contoh_before_after_segmentasi")
    args = ap.parse_args()

    entri_test = ambil_kelompok_test(args.csv, args.seed, args.rasio_train, args.rasio_valid)

    # Pilih contoh berimbang: separuh dari kelas 'sehat', separuh dari 'sakit'
    per_label = defaultdict(list)
    for e in entri_test:
        label = split_mod.normalisasi_label(e["state"])
        per_label[label].append(e)

    rng = random.Random(args.seed)
    for label in per_label:
        rng.shuffle(per_label[label])

    n_per_label = max(1, args.n_contoh // 2)
    terpilih = per_label.get("sehat", [])[:n_per_label] + per_label.get("sakit", [])[:n_per_label]

    if not terpilih:
        print("⚠️ Tidak ada entri test yang bisa dipakai -- cek --csv / --seed / rasio.")
        return

    os.makedirs(args.tujuan, exist_ok=True)
    print(f"\nMenyimpan {len(terpilih)} contoh before/after ke: {args.tujuan}")

    for entri in terpilih:
        src_path = os.path.join(args.photos_dir, entri["filename"])
        if not os.path.exists(src_path):
            print(f"  ⚠️ Lewati (file tidak ketemu): {src_path}")
            continue

        label = split_mod.normalisasi_label(entri["state"])
        img_before = Image.open(src_path).convert("RGB")
        img_after = split_mod.buat_versi_lab(img_before, entri["polygon"])

        nama_dasar = f"{label}_{entri['filename']}"
        img_before.save(os.path.join(args.tujuan, f"before_{nama_dasar}"), quality=95)
        img_after.save(os.path.join(args.tujuan, f"after_{nama_dasar}"), quality=95)

        gabungan = gabungkan_sisi_bersisi(img_before, img_after)
        gabungan.save(os.path.join(args.tujuan, f"gabungan_{nama_dasar}"), quality=95)

        print(f"  ✅ {entri['filename']} ({label})")

    print(f"\n🎉 Selesai. Untuk Gambar 4.2, tinggal pilih salah satu file 'gabungan_*.jpg'")
    print(f"   di folder '{args.tujuan}/' -- sudah berupa satu gambar sisi-bersisi,")
    print("   siap ditempel langsung ke Bab IV tanpa perlu edit tambahan.")


if __name__ == "__main__":
    main()
