"""
00_inspeksi_rocole.py
========================
Jalankan ini SEKALI setelah download & ekstrak RoCoLe, sebelum bikin apapun.
Tujuannya cuma satu: melihat struktur folder dan format anotasi yang
sebenarnya, supaya script pemisah domain (lab-crop vs field-full) bisa
dibuat presisi -- bukan tebak-tebakan.

Cara pakai:
    python 00_inspeksi_rocole.py --root ./RoCoLe
"""

import os
import argparse
import json


def tampilkan_pohon(root, max_depth=3, max_files_per_folder=5):
    for current_root, dirs, files in os.walk(root):
        depth = current_root[len(root):].count(os.sep)
        if depth > max_depth:
            dirs[:] = []
            continue
        indent = "  " * depth
        print(f"{indent}{os.path.basename(current_root)}/")
        for f in sorted(files)[:max_files_per_folder]:
            print(f"{indent}  {f}")
        if len(files) > max_files_per_folder:
            print(f"{indent}  ... (+{len(files) - max_files_per_folder} file lain)")


def cek_file_anotasi(root):
    print("\n=== Mencari kemungkinan file anotasi ===")
    ekstensi_anotasi = (".json", ".xml", ".csv", ".txt", ".xlsx")
    ditemukan = []
    for current_root, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith(ekstensi_anotasi):
                ditemukan.append(os.path.join(current_root, f))

    for fp in ditemukan[:10]:
        print(f"\n--- {fp} ---")
        try:
            if fp.endswith(".json"):
                data = json.load(open(fp, encoding="utf-8"))
                if isinstance(data, dict):
                    print("Top-level keys:", list(data.keys())[:10])
                    # kalau format COCO, tunjukkan contoh 1 annotation
                    if "annotations" in data and len(data["annotations"]) > 0:
                        print("Contoh 1 annotation:", json.dumps(data["annotations"][0], indent=2)[:500])
                    if "images" in data and len(data["images"]) > 0:
                        print("Contoh 1 image entry:", json.dumps(data["images"][0], indent=2)[:300])
                elif isinstance(data, list):
                    print("List dengan", len(data), "entri. Contoh entri pertama:")
                    print(json.dumps(data[0], indent=2)[:500])
            else:
                with open(fp, encoding="utf-8", errors="ignore") as fh:
                    isi = fh.read(1000)
                print(isi)
        except Exception as e:
            print(f"  (gagal dibaca otomatis: {e})")

    if len(ditemukan) > 10:
        print(f"\n... (+{len(ditemukan) - 10} file anotasi lain tidak ditampilkan)")

    if not ditemukan:
        print("Tidak ada file anotasi (.json/.xml/.csv/.txt/.xlsx) ditemukan.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    args = ap.parse_args()

    print(f"=== Struktur folder: {args.root} ===")
    tampilkan_pohon(args.root)

    cek_file_anotasi(args.root)

    print("\n=== SELESAI ===")
    print("Copy-paste SELURUH output di atas dan kirim ke saya -- dari situ")
    print("saya bisa langsung tulis script pemisah domain lab-vs-field yang presisi.")


if __name__ == "__main__":
    main()
