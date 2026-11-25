# Tennis Vision Datasets Collection & Guide

Direktori ini berisi kode, metadata, pretrained weights, dan panduan untuk mengunduh 4 dataset referensi:

---

## 1. TrackNet (Tennis Ball Tracking & Trajectory)
- **Status:** ✅ **Repository & Pretrained Model Terpasang**
- **Direktori:** `datasets/tracknet/`
- **File Model:** `models/tracknet.pt` (42.8 MB) & `datasets/tracknet/tracknet_weights.pt`
- **Arsitektur:** Menggunakan 3 frame berurutan untuk menghasilkan *heatmap* probabilitas posisi bola tenis berkecepatan tinggi.
- **Unduh Dataset Mentah Video (19.835 labeled frames):**
  Tersedia di Google Drive resmi penulis:
  `https://drive.google.com/drive/folders/11r0RUaQHX7I3ANkaYG4jOxXK1OYo01Ut`

---

## 2. Tennis Court Keypoints (Gholamreza)
- **Status:** ℹ️ **Siap Unduh (Ukuran: 7.25 GB)**
- **Sumber:** Hugging Face (`Gholamreza/tennis_court_keypoints_dataset`)
- **Direktori:** `datasets/court_keypoints/`
- **Fungsi:** Titik-titik sudut lapangan tenis untuk kalibrasi kamera dan homografi otomatis.
- **Cara Unduh:**
  Jalankan script di dalam foldernya:
  ```powershell
  python datasets/court_keypoints/download_court_dataset.py
  ```

---

## 3. Tennis Ball Detection (Roboflow Universe)
- **Status:** 🔑 **Membutuhkan API Key Gratis Roboflow / Download Langsung via Browser**
- **Sumber:** `https://universe.roboflow.com/teniss/tennis-ball-detection-rujjm-31q69`
- **Direktori:** `datasets/roboflow_ball_detection/`
- **Jumlah Data:** 5.743 gambar berlabel *tennis ball*.
- **Cara Unduh:**
  - **Opsi A (Browser - Direkomendasikan):** Buka link di browser, klik tombol **"Download Dataset"**, pilih format **YOLOv8**, lalu ekstrak zip ke folder `datasets/roboflow_ball_detection/`.
  - **Opsi B (CLI):** Masukkan API key gratis Anda ke `download_roboflow.py` lalu jalankan script.

---

## 4. OpenSportsLab Bounce & Action Localization (OSL-loc-tennis-public)
- **Status:** 🔒 **Gated Dataset di Hugging Face (Perlu Login Akun HF)**
- **Sumber:** `https://huggingface.co/datasets/OpenSportsLab/OSL-loc-tennis-public`
- **Direktori:** `datasets/osl_bounce_localization/`
- **Fungsi:** Anotasi kejadian pantulan bola (*bounce events*) dan aksi pertandingan.
- **Cara Unduh:**
  1. Kunjungi halaman dataset di Hugging Face: `https://huggingface.co/datasets/OpenSportsLab/OSL-loc-tennis-public`
  2. Klik tombol **"Agree and access repository"** (gratis).
  3. Jalankan script `python datasets/osl_bounce_localization/download_osl.py` dengan memasukkan Hugging Face Access Token Anda.
