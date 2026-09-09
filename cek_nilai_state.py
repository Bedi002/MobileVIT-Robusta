"""
cek_nilai_state.py
=====================
Diagnostik cepat: tampilkan semua nilai unik "state" yang benar-benar ada
di kolom Label RoCoLE-csv.csv, beserta berapa kali masing-masing muncul.
Ini buat mastiin logika normalisasi_label() di 01_split_domain_rocole.py
mencocokkan nilai yang tepat.

Cara pakai:
    python cek_nilai_state.py --csv "./Dataset/RoCoLe A robusta coffee leaf images dataset/Annotations/RoCoLE-csv.csv"
"""

import csv
import json
import argparse
from collections import Counter

csv.field_size_limit(10_000_000)

ap = argparse.ArgumentParser()
ap.add_argument("--csv", required=True)
args = ap.parse_args()

counter = Counter()
contoh_baris_mentah = None

with open(args.csv, encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        label_raw = row.get("Label", "")
        if not label_raw or label_raw.strip() in ("", "Skip"):
            continue
        try:
            label = json.loads(label_raw)
        except json.JSONDecodeError:
            continue

        if contoh_baris_mentah is None:
            contoh_baris_mentah = label_raw[:800]

        daun_list = label.get("Leaf", [])
        for daun in daun_list:
            state = daun.get("state", "<TIDAK ADA KEY 'state'>")
            counter[state] += 1

print("=== Nilai 'state' yang ditemukan (dan jumlah kemunculannya) ===")
for state, jumlah in counter.most_common():
    print(f"  {state!r}: {jumlah}")

print("\n=== Contoh isi mentah kolom Label (1 baris) ===")
print(contoh_baris_mentah)
