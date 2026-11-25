"""Download script for Tennis Court Keypoints Dataset from Hugging Face
Source: https://huggingface.co/datasets/Gholamreza/tennis_court_keypoints_dataset
File: tennis_court_det_dataset.zip (~7.25 GB)
"""
import os
import sys
import urllib.request
import zipfile

ZIP_URL = "https://huggingface.co/datasets/Gholamreza/tennis_court_keypoints_dataset/resolve/main/tennis_court_det_dataset.zip"
TARGET_DIR = os.path.dirname(os.path.abspath(__file__))
ZIP_FILE = os.path.join(TARGET_DIR, "tennis_court_det_dataset.zip")


def reporthook(blocknum, blocksize, totalsize):
    read = blocknum * blocksize
    if totalsize > 0:
        percent = read * 100 / totalsize
        mb_read = read / (1024 * 1024)
        mb_total = totalsize / (1024 * 1024)
        sys.stdout.write(f"\rDownloading: {mb_read:.1f} MB / {mb_total:.1f} MB ({percent:.1f}%)")
        sys.stdout.flush()


def download_dataset():
    print(f"Target URL: {ZIP_URL}")
    print(f"Saving to : {ZIP_FILE}")
    print("Ukuran file: ~7.25 GB. Pastikan koneksi internet stabil...")

    if os.path.exists(ZIP_FILE):
        print(f"File zip sudah ada: {ZIP_FILE}")
    else:
        urllib.request.urlretrieve(ZIP_URL, ZIP_FILE, reporthook)
        print("\nUnduhan selesai!")

    extract = input("Ekstrak file zip sekarang? (y/n): ").strip().lower()
    if extract == 'y':
        print("Mengekstrak file zip...")
        with zipfile.ZipFile(ZIP_FILE, 'r') as zip_ref:
            zip_ref.extractall(TARGET_DIR)
        print(f"Berhasil diekstrak ke {TARGET_DIR}")


if __name__ == "__main__":
    download_dataset()
