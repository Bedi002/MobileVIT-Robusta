import streamlit as st
import torch
import torch.nn as nn
import timm
import torchvision.transforms as T
from PIL import Image
import pandas as pd
import os # Tambahkan import os

# Dapatkan direktori tempat app.py berada (folder Streamlit)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Arahkan ke folder checkpoints (mundur 1 folder, lalu masuk ke 'checkpoints')
CHECKPOINT_DIR = os.path.join(BASE_DIR, "..", "checkpoints")

# Konfigurasi Halaman Streamlit
st.set_page_config(page_title="Klasifikasi Penyakit Daun Kopi", page_icon="🍃", layout="wide")

CLASSES = ["sakit", "sehat"]

# Perbarui path model agar mengarah ke folder checkpoints
MODEL_FILES = {
    "MobileNetV2": {
        "path": os.path.join(CHECKPOINT_DIR, "best_mobilenetv2_100_trainseed42_model.pth"),
        "backbone": "mobilenetv2_100",
        "features": 1280,
    },
    "MobileViT-S": {
        "path": os.path.join(CHECKPOINT_DIR, "best_mobilevit_s_trainseed42_model.pth"),
        "backbone": "mobilevit_s",
        "features": 640,
    },
}

TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class Classifier(nn.Module):
    """Backbone (timm, num_classes=0) + Linear head — sesuai struktur checkpoint asli."""
    def __init__(self, backbone_name, num_features):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=False, num_classes=0)
        self.fc = nn.Linear(num_features, 2)

    def forward(self, x):
        return self.fc(self.backbone(x))

@st.cache_resource
def load_models():
    """Fungsi di-cache agar model tidak diload ulang setiap kali halaman direfresh"""
    models = {}
    for name, cfg in MODEL_FILES.items():
        model = Classifier(cfg["backbone"], cfg["features"])
        try:
            state_dict = torch.load(cfg["path"], map_location="cpu", weights_only=False)
            model.load_state_dict(state_dict)
            model.eval()
            models[name] = model
        except FileNotFoundError:
            st.error(f"File model '{cfg['path']}' tidak ditemukan! Pastikan file sudah diunggah.")
    return models

# Tampilan Utama (Header & Deskripsi)
st.title("🍃 Klasifikasi Penyakit Daun Kopi Robusta")
st.markdown("""
Demo skripsi: **MobileViT vs MobileNetV2** untuk deteksi karat daun (*Hemileia vastatrix*) 
pada citra daun kopi robusta.

Unggah foto daun kopi (sehat atau sakit), lalu bandingkan prediksi kedua arsitektur.

> **Catatan:** demo ini hanya menjalankan tahap klasifikasi. Model segmentasi otomatis 
> (U-Net) tidak disertakan, sehingga foto dengan latar belakang alami/ramai bisa 
> menunjukkan akurasi lebih rendah — ini justru mendemonstrasikan langsung fenomena 
> *generalization gap* yang dibahas pada Bab IV skripsi.
""")

# Muat Model
models = load_models()

# Unggah Gambar
uploaded_file = st.file_uploader("Unggah Foto Daun Kopi", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption="Foto Daun Kopi yang Diunggah", use_container_width=True)
        
    with col2:
        st.subheader("Ringkasan Prediksi")
        
        if not models:
            st.warning("Model belum dimuat. Mohon cek file .pth Anda.")
        else:
            x = TRANSFORM(image.convert("RGB")).unsqueeze(0)
            
            for name, model in models.items():
                with torch.no_grad():
                    probs = torch.softmax(model(x), dim=1)[0]
                
                label_idx = int(probs.argmax())
                label = CLASSES[label_idx]
                conf = float(probs.max())
                
                # Tampilkan hasil teks
                st.markdown(f"#### **{name}**: {label.upper()} ({conf * 100:.1f}% yakin)")
                
                # Siapkan data untuk grafik bar probabilitas
                df_probs = pd.DataFrame({
                    "Kelas": CLASSES,
                    "Probabilitas": [float(probs[0]), float(probs[1])]
                }).set_index("Kelas")
                
                # Tampilkan grafik bar
                st.bar_chart(df_probs, height=150)

st.markdown("---")
st.markdown("Dataset: [RoCoLe (Robusta Coffee Leaf)](https://doi.org/10.1016/j.dib.2019.104414) — Parraga-Alava dkk. (2019)")