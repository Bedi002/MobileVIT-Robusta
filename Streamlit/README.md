---
title: Klasifikasi Penyakit Daun Kopi Robusta
emoji: 🍃
colorFrom: green
colorTo: yellow
sdk: gradio
sdk_version: 6.28.0
app_file: app.py
pinned: false
---

# Klasifikasi Penyakit Daun Kopi Robusta

Demo skripsi: **Klasifikasi Penyakit Daun Kopi Robusta Menggunakan Arsitektur MobileViT dengan
Pipeline Segmentasi Otomatis untuk Mengatasi Generalization Gap**

Membandingkan dua arsitektur klasifikasi (MobileNetV2 vs MobileViT-S) untuk mendeteksi karat
daun kopi (*Hemileia vastatrix*) pada citra daun kopi robusta, dilatih pada dataset
[RoCoLe](https://doi.org/10.1016/j.dib.2019.104414).

## Cara pakai
Unggah foto daun kopi, lalu lihat prediksi dan tingkat keyakinan dari kedua model.

## Catatan
Model segmentasi otomatis (U-Net) tidak disertakan pada demo ini — hanya tahap klasifikasi.
