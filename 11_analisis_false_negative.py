"""
11_analisis_false_negative.py
================================
Menjawab poin penyempurnaan #4: menyelidiki lebih dalam kasus daun SAKIT
yang justru diprediksi SEHAT (false negative) di pipeline deployment
(auto-segment U-Net + klasifikasi). Ini kasus paling berisiko secara
praktis -- petani bisa terlambat menangani penyakit karena dikira sehat.

Untuk SETIAP arsitektur classifier, script ini:
  1. Menjalankan pipeline auto-segment + klasifikasi penuh (seperti 06/08)
  2. Mengelompokkan hasil ke 4 kategori: TP, TN, FP (sehat dikira sakit --
     kurang berisiko, cuma bikin petani cek ulang tanpa perlu), dan
     FN (sakit dikira sehat -- PALING BERISIKO, prioritas analisis)
  3. Menyimpan SEMUA gambar hasil crop untuk kasus FN (dan FP untuk
     pembanding) ke folder terpisah per arsitektur, supaya bisa
     diinspeksi visual satu-satu
  4. Menulis metadata (nama file, arsitektur, kategori) ke CSV supaya
     mudah dirujuk saat menulis pembahasan di Bab IV

Cara pakai:
    python 11_analisis_false_negative.py \
        --csv "./Dataset/RoCoLe A robusta coffee leaf images dataset/Annotations/RoCoLE-csv.csv" \
        --photos_dir "./Dataset/RoCoLe A robusta coffee leaf images dataset/Photos" \
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
from torchvision import transforms
from importlib import import_module

split_mod = import_module("01_split_domain_rocole")
unet_mod = import_module("05_train_unet_segmentasi")
train_mod = import_module("02_train_mobilevit")
pipeline_mod = import_module("06_pipeline_deployment_real")

KELAS = ["sakit", "sehat"]


def kategorikan(label_asli, label_pred):
    if label_asli == "sakit" and label_pred == "sehat":
        return "FN_sakit_dikira_sehat"  # PALING BERISIKO
    if label_asli == "sehat" and label_pred == "sakit":
        return "FP_sehat_dikira_sakit"
    if label_asli == "sakit" and label_pred == "sakit":
        return "TP_benar_sakit"
    return "TN_benar_sehat"


def jalankan_dan_analisis(label, backbone, tag, unet, entri_test, photos_dir, device,
                           output_root, csv_writer):
    print(f"\n>> Menjalankan pipeline + analisis error untuk {label} (backbone={backbone}) ...")
    classifier = train_mod.DetectorModel(num_classes=2, backbone_name=backbone).to(device)
    classifier.load_state_dict(torch.load(f"checkpoints/best_{tag}_model.pth", map_location=device))
    classifier.eval()

    classify_tf = transforms.Compose([
        transforms.Resize((train_mod.IMG_SIZE, train_mod.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    hitung = {"FN_sakit_dikira_sehat": 0, "FP_sehat_dikira_sakit": 0,
              "TP_benar_sakit": 0, "TN_benar_sehat": 0}
    gagal_deteksi_fn = 0

    for entri in entri_test:
        img_path = os.path.join(photos_dir, entri["filename"])
        if not os.path.exists(img_path):
            continue
        img = Image.open(img_path).convert("RGB")
        label_asli = split_mod.normalisasi_label(entri["state"])

        hasil_crop, berhasil_deteksi = pipeline_mod.auto_segment_dan_crop(unet, img, device)

        x = classify_tf(hasil_crop.convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = classifier(x)
            pred_idx = torch.argmax(logits, dim=1).item()
        label_pred = KELAS[pred_idx]

        kategori = kategorikan(label_asli, label_pred)
        hitung[kategori] += 1

        # Simpan gambar untuk kategori paling penting: FN (prioritas) dan FP (pembanding)
        if kategori in ("FN_sakit_dikira_sehat", "FP_sehat_dikira_sakit"):
            if kategori == "FN_sakit_dikira_sehat" and not berhasil_deteksi:
                gagal_deteksi_fn += 1
            subfolder = os.path.join(output_root, label.replace(" ", "_"), kategori)
            os.makedirs(subfolder, exist_ok=True)
            nama_file = f"{kategori}_{entri['filename']}"
            hasil_crop.save(os.path.join(subfolder, nama_file))

            csv_writer.writerow({
                "arsitektur": label,
                "nama_file_asli": entri["filename"],
                "kategori": kategori,
                "segmentasi_terdeteksi": "gagal (pakai foto utuh)" if not berhasil_deteksi else "berhasil",
                "path_hasil_crop": os.path.join(subfolder, nama_file),
            })

    total = sum(hitung.values())
    print(f"   Total foto: {total}")
    for k, v in hitung.items():
        print(f"   {k}: {v} ({v/total*100:.1f}%)" if total else f"   {k}: {v}")
    print(f"   Dari kasus FN, {gagal_deteksi_fn} di antaranya juga gagal deteksi segmentasi U-Net sama sekali")
    print(f"   (foto utuh dipakai sebagai fallback -- kemungkinan penyebab tambahan kesalahan klasifikasi)")

    return hitung


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--photos_dir", required=True)
    ap.add_argument("--unet_checkpoint", default="checkpoints/best_unet_segmentasi.pth")
    ap.add_argument("--tag_mobilevit", default="mobilevit_labdomain")
    ap.add_argument("--tag_mobilenetv2", default="mobilenetv2_baseline")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rasio_train", type=float, default=0.70)
    ap.add_argument("--rasio_valid", type=float, default=0.15)
    ap.add_argument("--output_root", default="analisis_error_pipeline")
    ap.add_argument("--output_csv", default="metadata_error_pipeline.csv")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Perangkat aktif: {device}")

    unet = unet_mod.UNetRingan().to(device)
    unet.load_state_dict(torch.load(args.unet_checkpoint, map_location=device))
    unet.eval()

    entri_test = pipeline_mod.ambil_kelompok_test(args.csv, args.seed, args.rasio_train, args.rasio_valid)
    print(f"Jumlah foto field (test, ASING bagi semua model): {len(entri_test)}")

    model_configs = [
        ("MobileViT", "mobilevit_s", args.tag_mobilevit),
        ("MobileNetV2", "mobilenetv2_100", args.tag_mobilenetv2),
    ]

    os.makedirs(args.output_root, exist_ok=True)
    ringkasan = {}

    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["arsitektur", "nama_file_asli", "kategori",
                      "segmentasi_terdeteksi", "path_hasil_crop"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for label, backbone, tag in model_configs:
            hitung = jalankan_dan_analisis(label, backbone, tag, unet, entri_test,
                                            args.photos_dir, device, args.output_root, writer)
            ringkasan[label] = hitung

    print(f"\n{'='*70}")
    print("RINGKASAN PERBANDINGAN FALSE NEGATIVE ANTAR ARSITEKTUR")
    print(f"{'='*70}")
    print(f"{'Arsitektur':<15}{'FN (sakit->sehat)':<22}{'FP (sehat->sakit)':<22}")
    print("-" * 60)
    for label in ringkasan:
        fn = ringkasan[label]["FN_sakit_dikira_sehat"]
        fp = ringkasan[label]["FP_sehat_dikira_sakit"]
        print(f"{label:<15}{fn:<22}{fp:<22}")

    print(f"\n🎉 Selesai. Gambar hasil crop untuk kasus FN & FP tersimpan di: {args.output_root}/")
    print(f"   Metadata lengkap (nama file asli, kategori, status segmentasi) di: {args.output_csv}")
    print("\nLangkah selanjutnya (manual): buka folder FN tiap arsitektur, cek satu-satu:")
    print("  - Apakah mayoritas karena mask U-Net cacat (kepotong/kena bayangan)?")
    print("    -> kalau ya, ini mendukung argumen sensitivitas terhadap noise segmentasi")
    print("  - Atau crop-nya sudah bersih tapi gejala penyakitnya memang halus/ringan?")
    print("    -> kalau ya, ini keterbatasan model yang perlu diakui di Bab V")
    print("  Ingat: FN (sakit dikira sehat) SECARA PRAKTIS lebih berisiko dari FP,")
    print("  karena petani jadi telat menangani penyakit yang sebenarnya sudah ada --")
    print("  poin ini layak ditulis eksplisit di pembahasan implikasi Bab IV/V.")


if __name__ == "__main__":
    main()
