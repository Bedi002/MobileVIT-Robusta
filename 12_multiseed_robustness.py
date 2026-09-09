"""
12_multiseed_robustness.py
=============================
Menjawab poin penyempurnaan #2: menguji apakah temuan "ranking berbalik"
(MobileNetV2 unggul di field mentah, tapi MobileViT unggul setelah pipeline
segmentasi) itu KONSISTEN atau cuma kebetulan satu kali training -- dengan
mengulang training beberapa kali pakai seed BERBEDA.

PENTING soal seed: seed SPLIT DATA (kelompok pohon train/valid/test) TETAP
sama (--split_seed, default 42) di semua run -- supaya domain gap yang
dibandingkan tetap kontrol yang sama. Yang divariasikan HANYA seed training
(init bobot classifier head, urutan shuffle batch, augmentasi acak) lewat
--train_seeds. Ini satu-satunya script di antara semua poin penyempurnaan
yang BENAR-BENAR butuh training ulang -- karena itu jumlah seed dibuat kecil
(default 3) supaya tidak terlalu mahal untuk skripsi S1.

Untuk tiap kombinasi (arsitektur x seed training), script akan:
  1. Training dari awal (checkpoint disimpan dengan tag unik per seed)
  2. Evaluasi ke TEST field mentah
  3-*  (kalau --jalankan_pipeline diaktifkan) evaluasi juga ke pipeline
       auto-segment penuh, supaya ranking di skenario akhir ikut diuji

Di akhir, dicetak mean +/- std akurasi per arsitektur per skenario.

Cara pakai (contoh minimal, cuma TEST field mentah):
    python 12_multiseed_robustness.py \
        --data_dir ./DATASET/rocole_domain \
        --train_seeds 42 123 2024 \
        --robust_aug

Cara pakai (lengkap, termasuk pipeline auto-segment -- lebih lama):
    python 12_multiseed_robustness.py \
        --data_dir ./DATASET/rocole_domain \
        --csv "./Dataset/RoCoLe A robusta coffee leaf images dataset/Annotations/RoCoLE-csv.csv" \
        --photos_dir "./Dataset/RoCoLe A robusta coffee leaf images dataset/Photos" \
        --unet_checkpoint checkpoints/best_unet_segmentasi.pth \
        --train_seeds 42 123 2024 \
        --robust_aug --jalankan_pipeline
"""

import os
import csv
import argparse
import random
import statistics
import torch
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from importlib import import_module

train_mod = import_module("02_train_mobilevit")


def train_satu_run(data_dir, backbone, train_seed, robust_aug, device):
    """Reimplementasi inti dari main() di 02_train_mobilevit.py, tapi dengan
    seed training yang bisa divariasikan TANPA mengubah seed split data
    (karena split data sudah difiksasi sebelumnya oleh 01_split_domain_rocole.py
    dan tidak disentuh lagi di sini)."""
    torch.manual_seed(train_seed)
    random.seed(train_seed)

    loaders, sizes, classes = train_mod.build_dataloaders(data_dir, robust_aug=robust_aug)
    model = train_mod.DetectorModel(num_classes=len(classes), backbone_name=backbone).to(device)

    tag = f"{backbone}_trainseed{train_seed}"
    train_mod.train(model, loaders, sizes, device, tag=tag, backbone_name=backbone, use_mlflow=False)
    return tag, model


def evaluate_imagefolder(model, folder_path, tf, device):
    ds = datasets.ImageFolder(folder_path, tf)
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=2)
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.numpy())
            y_pred.extend(preds.cpu().numpy())
    return accuracy_score(y_true, y_pred)


def evaluate_pipeline_acc(model, unet, entri_test, photos_dir, tf, device, split_mod, pipeline_mod):
    KELAS = ["sakit", "sehat"]
    model.eval()
    y_true, y_pred = [], []
    for entri in entri_test:
        img_path = os.path.join(photos_dir, entri["filename"])
        if not os.path.exists(img_path):
            continue
        img = Image.open(img_path).convert("RGB")
        label_asli = split_mod.normalisasi_label(entri["state"])
        hasil_crop, _ = pipeline_mod.auto_segment_dan_crop(unet, img, device)
        x = tf(hasil_crop.convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            pred_idx = torch.argmax(model(x), dim=1).item()
        y_true.append(label_asli)
        y_pred.append(KELAS[pred_idx])
    return accuracy_score(y_true, y_pred)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--train_seeds", type=int, nargs="+", default=[42, 123, 2024])
    ap.add_argument("--robust_aug", action="store_true")
    ap.add_argument("--jalankan_pipeline", action="store_true",
                     help="Kalau aktif, tiap run juga dievaluasi ke pipeline auto-segment penuh "
                          "(butuh --csv, --photos_dir, --unet_checkpoint). Lebih lama.")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--photos_dir", default=None)
    ap.add_argument("--unet_checkpoint", default="checkpoints/best_unet_segmentasi.pth")
    ap.add_argument("--split_seed", type=int, default=42,
                     help="Seed split data -- HARUS sama dengan yang dipakai 01_split_domain_rocole.py, "
                          "jangan diubah kecuali kamu juga menjalankan ulang split-nya.")
    ap.add_argument("--rasio_train", type=float, default=0.70)
    ap.add_argument("--rasio_valid", type=float, default=0.15)
    ap.add_argument("--output_csv", default="hasil_multiseed_robustness.csv")
    args = ap.parse_args()

    if args.jalankan_pipeline and not (args.csv and args.photos_dir):
        ap.error("--jalankan_pipeline butuh --csv dan --photos_dir")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Perangkat aktif: {device}")
    print(f"Seed training yang diuji: {args.train_seeds}")
    print(f"Seed split data (TETAP, tidak diubah): {args.split_seed}")

    unet = None
    entri_test = None
    split_mod = pipeline_mod = None
    if args.jalankan_pipeline:
        split_mod = import_module("01_split_domain_rocole")
        unet_mod = import_module("05_train_unet_segmentasi")
        pipeline_mod = import_module("06_pipeline_deployment_real")
        unet = unet_mod.UNetRingan().to(device)
        unet.load_state_dict(torch.load(args.unet_checkpoint, map_location=device))
        unet.eval()
        entri_test = pipeline_mod.ambil_kelompok_test(args.csv, args.split_seed,
                                                        args.rasio_train, args.rasio_valid)

    eval_tf = transforms.Compose([
        transforms.Resize((train_mod.IMG_SIZE, train_mod.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    arsitektur_configs = [("MobileViT", "mobilevit_s"), ("MobileNetV2", "mobilenetv2_100")]
    hasil = {label: {"test_field": [], "pipeline": []} for label, _ in arsitektur_configs}

    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["arsitektur", "train_seed", "akurasi_test_field", "akurasi_pipeline"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for label, backbone in arsitektur_configs:
            for train_seed in args.train_seeds:
                print(f"\n{'='*70}\n{label} | train_seed={train_seed}\n{'='*70}")
                tag, model = train_satu_run(args.data_dir, backbone, train_seed, args.robust_aug, device)

                acc_field = evaluate_imagefolder(model, f"{args.data_dir}/test", eval_tf, device)
                print(f"   Akurasi TEST field mentah (train_seed={train_seed}): {acc_field*100:.2f}%")
                hasil[label]["test_field"].append(acc_field)

                acc_pipe = None
                if args.jalankan_pipeline:
                    acc_pipe = evaluate_pipeline_acc(model, unet, entri_test, args.photos_dir,
                                                      eval_tf, device, split_mod, pipeline_mod)
                    print(f"   Akurasi PIPELINE deployment (train_seed={train_seed}): {acc_pipe*100:.2f}%")
                    hasil[label]["pipeline"].append(acc_pipe)

                writer.writerow({
                    "arsitektur": label,
                    "train_seed": train_seed,
                    "akurasi_test_field": f"{acc_field:.4f}",
                    "akurasi_pipeline": f"{acc_pipe:.4f}" if acc_pipe is not None else "",
                })

    print(f"\n{'='*70}\nRINGKASAN MEAN +/- STD ANTAR SEED TRAINING\n{'='*70}")
    for label in hasil:
        vals_field = hasil[label]["test_field"]
        mean_f = statistics.mean(vals_field) * 100
        std_f = statistics.stdev(vals_field) * 100 if len(vals_field) > 1 else 0.0
        print(f"{label:<15} TEST field mentah : {mean_f:.2f}% +/- {std_f:.2f}% (n={len(vals_field)} run)")

        if args.jalankan_pipeline:
            vals_pipe = hasil[label]["pipeline"]
            mean_p = statistics.mean(vals_pipe) * 100
            std_p = statistics.stdev(vals_pipe) * 100 if len(vals_pipe) > 1 else 0.0
            print(f"{label:<15} PIPELINE deploy    : {mean_p:.2f}% +/- {std_p:.2f}% (n={len(vals_pipe)} run)")

    print(f"\n🎉 Selesai. Detail per-run tersimpan di: {args.output_csv}")
    print("\nCara baca hasil untuk Bab IV:")
    print("- Kalau interval mean+/-std TEST field kedua arsitektur TIDAK saling")
    print("  tumpang tindih (mis. MobileNetV2 konsisten di atas MobileViT di semua")
    print("  seed), klaim 'ranking berbalik' didukung kuat -- bukan kebetulan.")
    print("- Kalau ada tumpang tindih, perlu kalimat yang lebih hati-hati di Bab IV/V,")
    print("  mis. 'kecenderungan konsisten' bukan 'MobileNetV2 pasti lebih general'.")
    print("- Checkpoint tiap run tersimpan di checkpoints/best_<backbone>_trainseed<N>_model.pth")
    print("  -- bisa dipakai ulang untuk analisis lain tanpa training lagi.")


if __name__ == "__main__":
    main()
