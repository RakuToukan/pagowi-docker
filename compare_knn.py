"""compare_knn.py - Bukti KNN NumPy (mosaic_core) sama dengan KNN loop asli (main_original.py).

Jalankan: python compare_knn.py
"""
import time

import numpy as np
from PIL import Image

import main_original as orig
import mosaic_core as core

GRID = 32
img = Image.open("sample.png").convert("RGB")
avg, *_ = core.get_avg_rgb_per_grid(img, GRID, GRID)

ok = True
for distance, fn in [("euclidean", orig.knn_euclidean),
                     ("minkowski", lambda t, d: orig.knn_minkowski(t, d, p=3)),
                     ("manhattan", orig.knn_manhattan)]:
    for category, folder in core.FOLDER_CATEGORY.items():
        data, _ = core.get_folder_avg_rgb(folder)
        t0 = time.time(); a = fn(avg, data); t1 = time.time()
        b = core.knn_grid(avg, data, distance); t2 = time.time()
        same = np.array_equal(a, b)
        ok &= same
        print(f"{distance:10s} {category:9s} sama={same}  loop={t1 - t0:6.2f}s  numpy={t2 - t1:5.3f}s")
print("SEMUA SAMA" if ok else "ADA PERBEDAAN")
