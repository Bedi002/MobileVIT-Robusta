"""
10_ukur_latency.py
=====================
Menjawab poin penyempurnaan #3: mengukur waktu inferensi (latency) dan
throughput (FPS) -- bukan cuma jumlah parameter -- untuk MobileViT,
MobileNetV2, dan U-Net segmentasi. Juga menghitung estimasi latency
TOTAL pipeline deployment (U-Net + classifier berurutan), karena itu
yang sebenarnya dirasakan pengguna aplikasi.

TIDAK PERLU TRAINING ULANG -- cukup checkpoint yang sudah ada.

Diukur di CPU (representatif untuk device petani/lapangan yang mungkin
tidak punya GPU) dan di GPU kalau tersedia, supaya klaim efisiensi di
Bab I bisa dipertanggungjawabkan dengan angka nyata, bukan cuma jumlah
parameter (yang justru menunjukkan MobileViT lebih besar dari
MobileNetV2: 4,9 juta vs 2,2 juta).

Cara pakai:
    python 10_ukur_latency.py \
        --tag_mobilevit mobilevit_labdomain \
        --tag_mobilenetv2 mobilenetv2_baseline \
        --unet_checkpoint checkpoints/best_unet_segmentasi.pth \
        --n_iter 100
"""

import argparse
import time
import torch
from importlib import import_module

train_mod = import_module("02_train_mobilevit")
unet_mod = import_module("05_train_unet_segmentasi")


def ukur_latency(forward_fn, dummy_input, device, n_warmup=20, n_iter=100):
    with torch.no_grad():
        for _ in range(n_warmup):
            forward_fn(dummy_input)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(n_iter):
            forward_fn(dummy_input)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
    latency_ms = (elapsed / n_iter) * 1000
    fps = 1000.0 / latency_ms
    return latency_ms, fps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag_mobilevit", default="mobilevit_labdomain")
    ap.add_argument("--tag_mobilenetv2", default="mobilenetv2_baseline")
    ap.add_argument("--unet_checkpoint", default="checkpoints/best_unet_segmentasi.pth")
    ap.add_argument("--n_warmup", type=int, default=20)
    ap.add_argument("--n_iter", type=int, default=100)
    args = ap.parse_args()

    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))

    hasil = {}  # {(nama_model, device_str): (latency_ms, fps)}

    classifier_configs = [
        ("MobileViT", "mobilevit_s", args.tag_mobilevit),
        ("MobileNetV2", "mobilenetv2_100", args.tag_mobilenetv2),
    ]

    for device in devices:
        print(f"\n{'='*60}\nPerangkat: {device}\n{'='*60}")

        # --- Classifier ---
        for label, backbone, tag in classifier_configs:
            model = train_mod.DetectorModel(num_classes=2, backbone_name=backbone).to(device)
            model.load_state_dict(torch.load(f"checkpoints/best_{tag}_model.pth", map_location=device))
            model.eval()
            dummy = torch.randn(1, 3, train_mod.IMG_SIZE, train_mod.IMG_SIZE).to(device)
            lat, fps = ukur_latency(model, dummy, device, args.n_warmup, args.n_iter)
            hasil[(label, str(device))] = (lat, fps)
            print(f"  {label:<15} : {lat:7.2f} ms/gambar | {fps:7.2f} FPS")

        # --- U-Net segmentasi ---
        unet = unet_mod.UNetRingan().to(device)
        unet.load_state_dict(torch.load(args.unet_checkpoint, map_location=device))
        unet.eval()
        dummy_unet = torch.randn(1, 3, unet_mod.IMG_SIZE, unet_mod.IMG_SIZE).to(device)
        lat_unet, fps_unet = ukur_latency(unet, dummy_unet, device, args.n_warmup, args.n_iter)
        hasil[("U-Net segmentasi", str(device))] = (lat_unet, fps_unet)
        print(f"  {'U-Net segmentasi':<15} : {lat_unet:7.2f} ms/gambar | {fps_unet:7.2f} FPS")

    print(f"\n{'='*70}")
    print("RINGKASAN LATENCY TOTAL PIPELINE DEPLOYMENT (U-Net + Classifier)")
    print("(estimasi kasar: latency U-Net + latency classifier, dijalankan berurutan;")
    print(" belum termasuk overhead crop/resize PIL yang relatif kecil)")
    print(f"{'='*70}")
    for device in devices:
        dstr = str(device)
        lat_unet = hasil[("U-Net segmentasi", dstr)][0]
        for label, _, _ in classifier_configs:
            lat_clf = hasil[(label, dstr)][0]
            lat_total = lat_unet + lat_clf
            fps_total = 1000.0 / lat_total
            print(f"  [{dstr}] {label:<12} + U-Net : {lat_total:7.2f} ms/gambar total "
                  f"| ~{fps_total:5.2f} FPS end-to-end")

    print("\nCatatan interpretasi untuk Bab IV/V:")
    print("- Kalau MobileViT lebih lambat dari MobileNetV2 meski akurasi pipeline")
    print("  lebih tinggi, ini trade-off yang jujur untuk dilaporkan -- bukan")
    print("  disembunyikan. Klaim 'efisien' sebaiknya diberi konteks: efisien")
    print("  dalam hal AKURASI DEPLOYMENT per parameter, bukan tercepat secara mutlak.")
    print("- Kalau target akhirnya aplikasi mobile/edge offline, angka CPU-lah yang")
    print("  paling relevan dilaporkan (bukan GPU), karena itu skenario realistis")
    print("  perangkat petani di lapangan.")


if __name__ == "__main__":
    main()
