"""Script untuk mengunduh OpenSportsLab Tennis Localization Dataset dari Hugging Face
Dataset: OpenSportsLab/OSL-loc-tennis-public
URL: https://huggingface.co/datasets/OpenSportsLab/OSL-loc-tennis-public

Catatan:
Dataset ini adalah Gated Dataset di Hugging Face.
Sebelum mengunduh:
1. Kunjungi https://huggingface.co/datasets/OpenSportsLab/OSL-loc-tennis-public
2. Login dan klik tombol 'Agree and access repository'
3. Buat Access Token di https://huggingface.co/settings/tokens (pilih Read)
4. Jalankan script ini: python download_osl.py
"""
import os
import sys

DATASET_REPO = "OpenSportsLab/OSL-loc-tennis-public"


def main():
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("==================================================================")
        print("Dataset OpenSportsLab membutuhkan akses Hugging Face Token.")
        print("Pastikan Anda sudah menyetujui lisensi di:")
        print("https://huggingface.co/datasets/OpenSportsLab/OSL-loc-tennis-public")
        print("==================================================================")
        token = input("Masukkan Hugging Face Token Anda (atau tekan Enter untuk batal): ").strip()

    if not token:
        print("Batal mengunduh. Dapatkan token di https://huggingface.co/settings/tokens")
        return

    try:
        from datasets import load_dataset
    except ImportError:
        print("Menginstal pustaka datasets...")
        os.system("pip install datasets huggingface_hub")
        from datasets import load_dataset

    target_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Memuat dataset {DATASET_REPO}...")
    try:
        ds = load_dataset(DATASET_REPO, token=token, cache_dir=target_dir)
        print("Dataset berhasil diunduh:")
        print(ds)
    except Exception as e:
        print(f"Gagal mengunduh dataset: {e}")


if __name__ == "__main__":
    main()
