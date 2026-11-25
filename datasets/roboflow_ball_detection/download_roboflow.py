"""Script untuk mengunduh dataset Tennis Ball Detection dari Roboflow Universe
URL: https://universe.roboflow.com/teniss/tennis-ball-detection-rujjm-31q69

Cara mendapatkan Roboflow API Key gratis:
1. Buka https://app.roboflow.com lalu login/daftar akun gratis.
2. Buka menu Settings -> Workspace API Key -> Copy API Key.
3. Jalankan script ini: python download_roboflow.py
"""
import os
import sys

ROBOFLOW_WORKSPACE = "teniss"
ROBOFLOW_PROJECT = "tennis-ball-detection-rujjm-31q69"
VERSION = 1
FORMAT = "yolov8"


def main():
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        print("================================================================")
        print("Roboflow API Key diperlukan untuk mengunduh dataset via CLI.")
        print("Alternatif lebih mudah: Anda dapat langsung membuka browser ke:")
        print("https://universe.roboflow.com/teniss/tennis-ball-detection-rujjm-31q69")
        print("Lalu klik 'Download Dataset' -> pilih format YOLOv8 -> unduh zip.")
        print("================================================================")
        api_key = input("Masukkan Roboflow API Key Anda (atau tekan Enter untuk batal): ").strip()

    if not api_key:
        print("Batal mengunduh. Silakan gunakan opsi unduh langsung via browser.")
        return

    try:
        from roboflow import Roboflow
    except ImportError:
        print("Menginstal pustaka roboflow...")
        os.system("pip install roboflow")
        from roboflow import Roboflow

    target_dir = os.path.dirname(os.path.abspath(__file__))
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(ROBOFLOW_WORKSPACE).project(ROBOFLOW_PROJECT)
    version = project.version(VERSION)
    print(f"Mengunduh dataset {ROBOFLOW_PROJECT} versi {VERSION} (format {FORMAT})...")
    version.download(FORMAT, location=target_dir)
    print("Download selesai!")


if __name__ == "__main__":
    main()
