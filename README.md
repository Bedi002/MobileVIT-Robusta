# Klasifikasi Penyakit Daun Kopi Robusta dengan MobileViT dan Pipeline Segmentasi Otomatis

Klasifikasi citra daun kopi robusta (sehat/sakit) menggunakan arsitektur hybrid CNN-Transformer **MobileViT**, dikombinasikan dengan pipeline **segmentasi otomatis berbasis U-Net** sebagai pra-pemrosesan, untuk mengatasi *generalization gap* antara kondisi laboratorium dan kondisi lapangan.

> Skripsi — Program Studi Informatika, Fakultas Ilmu Komputer, Universitas AMIKOM Yogyakarta (2026)

---

## Daftar Isi

- [Latar Belakang](#latar-belakang)
- [Fitur Utama](#fitur-utama)
- [Arsitektur](#arsitektur)
- [Dataset](#dataset)
- [Struktur Proyek](#struktur-proyek)
- [Instalasi](#instalasi)
- [Cara Penggunaan](#cara-penggunaan)
- [Hasil Eksperimen](#hasil-eksperimen)
- [Tech Stack](#tech-stack)
- [Referensi](#referensi)
- [Penulis](#penulis)

---

## Latar Belakang

Deteksi dini penyakit karat daun (*coffee leaf rust*) pada kopi robusta penting untuk mencegah penurunan hasil panen, namun keterbatasan tenaga penyuluh membuat identifikasi manual sulit dilakukan secara masif. Banyak penelitian klasifikasi penyakit daun kopi berbasis deep learning melaporkan akurasi tinggi, tetapi umumnya dievaluasi dengan skema pembagian data acak (*random split*) yang rawan kebocoran data dan tidak mengukur performa model pada kondisi lapangan yang sesungguhnya — fenomena yang dikenal sebagai ***generalization gap***.

Proyek ini mengusulkan pendekatan yang:
1. Menggunakan **MobileViT** — arsitektur hybrid CNN-Transformer yang ringan namun mampu menangkap representasi lokal sekaligus global.
2. Menambahkan **U-Net** sebagai pipeline segmentasi otomatis untuk menghilangkan latar belakang sebelum klasifikasi.
3. Mengevaluasi generalisasi model secara eksplisit menggunakan **pembagian data berbasis kelompok pohon** (bukan per-citra) dan skema **domain laboratorium vs domain lapangan**.

## Fitur Utama

- Pipeline end-to-end: segmentasi otomatis → klasifikasi, dievaluasi pada citra lapangan yang belum pernah dilihat model.
- Perbandingan langsung MobileViT vs MobileNetV2 pada skema evaluasi yang identik.
- Pengukuran *generalization gap* eksplisit antara domain lab dan domain lapangan.
- Skema pembagian data berbasis kelompok pohon untuk mencegah kebocoran data.
- Tracking eksperimen menggunakan MLflow/DagsHub.

## Arsitektur

**Blok MobileViT** bekerja melalui tiga tahap:
1. **Representasi lokal** — konvolusi n×n + 1×1 (khas CNN).
2. **Representasi global** — feature map dipecah (*unfold*) menjadi patch, diproses Transformer (*self-attention*).
3. **Fusion** — hasil global dikembalikan ke bentuk spasial (*fold*) dan digabung dengan representasi lokal.

**U-Net** digunakan sebagai model segmentasi terpisah (encoder-decoder dengan skip connection) yang memisahkan objek daun dari latar belakang sebelum citra masuk ke tahap klasifikasi.

Detail arsitektur, diagram, dan seluruh persamaan matematis (konvolusi, self-attention, loss function, metrik evaluasi) didokumentasikan lengkap pada `docs/BAB_2_Dasar_Teori.docx`.

## Dataset

Menggunakan dataset publik **[RoCoLe](https://data.mendeley.com/datasets/c5yvn32dzg/2) (Robusta Coffee Leaf)** — 1.560 citra daun kopi robusta dengan anotasi status kesehatan (sehat/sakit) dan anotasi *polygon* segmentasi daun.

Pembagian data berbasis kelompok pohon/kebun asal citra (bukan per-citra) untuk mencegah kebocoran data:

| Subset | Proporsi | Jumlah Citra | Domain |
|---|---|---|---|
| Data Latih | 70% | 1.092 | Laboratorium (crop + latar putih) |
| Data Validasi | 15% | 232 | Laboratorium (crop + latar putih) |
| Data Uji | 15% | 236 | Lapangan (citra asli, tanpa olahan) |

## Struktur Proyek

```
├── data/
│   ├── raw/                 # Dataset RoCoLe asli (citra + anotasi)
│   └── processed/           # Hasil split kelompok pohon + domain lab/lapangan
├── src/
│   ├── segmentation/        # Model & training U-Net
│   ├── classification/      # Model & training MobileViT / MobileNetV2
│   ├── data_split.py        # Skrip pembagian data berbasis kelompok pohon
│   └── pipeline.py          # Pipeline end-to-end (segmentasi -> klasifikasi)
├── notebooks/                # Notebook eksperimen & eksplorasi
├── results/
│   ├── metrics/              # Hasil evaluasi (accuracy, precision, recall, F1, IoU)
│   └── figures/               # Grafik & visualisasi hasil
├── docs/                      # Dokumen skripsi (Bab 1-4, referensi)
├── requirements.txt
└── README.md
```

> Sesuaikan struktur di atas dengan struktur folder repo yang sebenarnya bila ada perbedaan.

## Instalasi

```bash
git clone https://github.com/Bedi002/MobileVIT-Robusta.git
cd MobileVIT-Robusta
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Cara Penggunaan

**1. Pembagian data berbasis kelompok pohon**
```bash
python src/data_split.py --data-dir data/raw --output-dir data/processed
```

**2. Melatih model segmentasi (U-Net)**
```bash
python src/segmentation/train.py --data-dir data/processed --epochs 50
```

**3. Melatih model klasifikasi (MobileViT)**
```bash
python src/classification/train.py --model mobilevit --data-dir data/processed --epochs 50
```

**4. Evaluasi pipeline end-to-end pada domain lapangan**
```bash
python src/pipeline.py --checkpoint-seg <path> --checkpoint-cls <path> --eval-domain field
```

## Hasil Eksperimen

| Model | Akurasi (Lab) | Akurasi (Lapangan) | Generalization Gap | Precision | Recall | F1-Score |
|---|---|---|---|---|---|---|
| MobileNetV2 (baseline) | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| MobileViT (tanpa segmentasi) | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| MobileViT + U-Net (pipeline lengkap) | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

**Segmentasi (U-Net):** IoU = _TBD_

> Isi tabel di atas dengan angka hasil eksperimen sebenarnya sebelum di-push ke GitHub.

## Tech Stack

- **Bahasa:** Python 3
- **Deep Learning:** PyTorch, Torchvision
- **Model:** timm (PyTorch Image Models) — MobileViT & MobileNetV2 pretrained
- **Eksperimen tracking:** MLflow, DagsHub
- **Evaluasi:** scikit-learn
- **Perangkat keras:** NVIDIA GeForce GTX 1660 Super (VRAM 6GB)

## Referensi

Daftar pustaka lengkap (16 referensi) tersedia pada `docs/BAB_2_Dasar_Teori.docx`. Rujukan utama:

- Mehta & Rastegari, "MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer," ICLR 2022.
- Ronneberger, Fischer & Brox, "U-Net: Convolutional Networks for Biomedical Image Segmentation," MICCAI 2015.
- Parraga-Alava et al., "RoCoLe: A Robusta Coffee Leaf Images Dataset," Data in Brief, 2019.

## Penulis

**Abdi Wicaksono B.S.**
NIM 23.11.5617 — Program Studi Informatika
Fakultas Ilmu Komputer, Universitas AMIKOM Yogyakarta

---

<sub>README ini dibuat berdasarkan dokumentasi Bab 1-3 skripsi. Sesuaikan bagian struktur proyek, cara penggunaan, dan hasil eksperimen dengan kondisi repo yang sebenarnya sebelum di-upload.</sub>
