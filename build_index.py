"""build_index.py - Membangun indeks KNN: rata-rata RGB tiap tile per kategori.

KNN tidak "belajar"; tahap ini hanya menyimpan data tile ke model.pkl
supaya API tidak membaca ribuan gambar setiap kali start/request.
"""
import pickle
import sys

from mosaic_core import FOLDER_CATEGORY, get_folder_avg_rgb

out_path = sys.argv[1] if len(sys.argv) > 1 else "model.pkl"
index = {}
for category, folder in FOLDER_CATEGORY.items():
    avg_rgb, paths = get_folder_avg_rgb(folder)
    if len(paths) == 0:
        sys.exit(f"Tidak ada tile di {folder}")
    index[category] = {"avg_rgb": avg_rgb, "paths": paths}
    print(f"{category:9s}: {len(paths)} tile")

with open(out_path, "wb") as f:
    pickle.dump(index, f)
print(f"Indeks tersimpan di {out_path}")
