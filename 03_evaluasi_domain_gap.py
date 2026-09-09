"""
03_evaluasi_domain_gap.py
============================
Evaluasi model MobileViT yang sudah dilatih (domain LAB) terhadap test set
domain FIELD (foto asli utuh, background alami, tidak pernah dilihat model
sama sekali selama training). Ini pengujian inti untuk menjawab pertanyaan
generalization gap pada skripsi cadangan ini.

Cara pakai:
    python 03_evaluasi_domain_gap.py --data_dir ./DATASET/rocole_domain --tag mobilevit_labdomain
"""

import argparse
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from importlib import import_module
train_mod = import_module("02_train_mobilevit")


def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.numpy())
            y_pred.extend(preds.cpu().numpy())
    return y_true, y_pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--tag", default="mobilevit_labdomain")
    ap.add_argument("--backbone", default="mobilevit_s",
                     help="Harus SAMA dengan --backbone yang dipakai saat training checkpoint ini")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    eval_tf = transforms.Compose([
        transforms.Resize((train_mod.IMG_SIZE, train_mod.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    model = train_mod.DetectorModel(num_classes=2, backbone_name=args.backbone).to(device)
    model.load_state_dict(torch.load(f"checkpoints/best_{args.tag}_model.pth", map_location=device))
    print(f"Backbone: {args.backbone} | Tag: {args.tag}")

    print("=" * 70)
    for split, label_domain in [("valid", "LAB (domain sama dengan training)"),
                                 ("test", "FIELD (domain ASING, belum pernah dilihat)")]:
        ds = datasets.ImageFolder(f"{args.data_dir}/{split}", eval_tf)
        loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=2)
        y_true, y_pred = evaluate(model, loader, device)
        acc = accuracy_score(y_true, y_pred)

        print(f"\n=== {split.upper()} -- {label_domain} ===")
        print(f"Jumlah gambar: {len(ds)} | Kelas: {ds.classes}")
        print(f"Akurasi: {acc*100:.2f}%")
        print(classification_report(y_true, y_pred, target_names=ds.classes))
        print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))

    print("=" * 70)
    print("\nCatatan: kalau akurasi TEST (field) jauh lebih rendah dari VALID (lab),")
    print("itu bukti generalization gap -- model belajar mengenali kondisi 'bersih'")
    print("tapi belum tentu belajar pola penyakit yang sesungguhnya, sama seperti")
    print("temuan lintas-generator di skripsi utama (sink label phenomenon).")


if __name__ == "__main__":
    main()
