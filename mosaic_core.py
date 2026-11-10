"""mosaic_core.py - Fungsi inti photomosaic dengan KNN (k=1).

Logika diambil dari main.py project uas_kdka. Perbedaan:
- KNN ditulis dengan NumPy (broadcasting), bukan 3 loop Python.
- Path tile memakai '/' agar jalan di Linux (container).
"""
import os

import numpy as np
from PIL import Image

FOLDER_CATEGORY = {
    "building": "assets/Building",
    "cloud": "assets/Cloud",
    "forest": "assets/Forest",
    "mountain": "assets/Mountain",
}
DISTANCES = ["euclidean", "minkowski", "manhattan"]
MINKOWSKI_P = 3


def get_folder_avg_rgb(folder):
    """Rata-rata R,G,B tiap gambar di folder. Return (array (N,3), list path)."""
    avg_rgb, img_paths = [], []
    for file in sorted(os.listdir(folder)):
        file_path = folder.rstrip("/") + "/" + file
        if not os.path.isfile(file_path):
            continue
        try:
            with Image.open(file_path) as img:
                arr = np.array(img.convert("RGB"))
                avg_rgb.append(arr.mean(axis=(0, 1)))
                img_paths.append(file_path)
        except Exception as e:  # file bukan gambar: lewati
            print(f"gagal: {file} = {e}")
    return np.array(avg_rgb), img_paths


def get_avg_rgb_per_grid(img, m, n):
    """Bagi gambar jadi grid m kolom x n baris, hitung rata-rata RGB tiap kotak."""
    img = img.convert("RGB")
    width, height = img.size
    grid_w, grid_h = width / m, height / n
    arr = np.array(img)
    out = np.zeros((n, m, 3))
    for row in range(n):
        for col in range(m):
            x1, x2 = int(col * grid_w), int((col + 1) * grid_w)
            y1, y2 = int(row * grid_h), int((row + 1) * grid_h)
            out[row, col] = arr[y1:y2, x1:x2].mean(axis=(0, 1))
    return out, width, height, grid_w, grid_h


def distance_matrix(targets, dataset, distance):
    """Jarak tiap target (T,3) ke tiap tile dataset (N,3) -> matriks (T,N)."""
    diff = np.abs(targets[:, None, :] - dataset[None, :, :])
    if distance == "euclidean":
        return np.sqrt((diff ** 2).sum(axis=2))
    if distance == "minkowski":
        return (diff ** MINKOWSKI_P).sum(axis=2) ** (1 / MINKOWSKI_P)
    if distance == "manhattan":
        return diff.sum(axis=2)
    raise ValueError(f"distance tidak dikenal: {distance}. Pilihan: {DISTANCES}")


def knn_nearest(targets, dataset, distance, chunk=1024):
    """Indeks tile terdekat (k=1) untuk tiap target. Return (indeks (T,), jarak (T,))."""
    idx = np.empty(len(targets), dtype=int)
    dist = np.empty(len(targets))
    for s in range(0, len(targets), chunk):
        d = distance_matrix(targets[s:s + chunk], dataset, distance)
        i = d.argmin(axis=1)
        idx[s:s + chunk] = i
        dist[s:s + chunk] = d[np.arange(len(i)), i]
    return idx, dist


def knn_grid(avg_color_pixel, dataset, distance):
    """KNN untuk seluruh grid (n,m,3) -> matriks indeks tile (n,m)."""
    n, m, _ = avg_color_pixel.shape
    idx, _ = knn_nearest(avg_color_pixel.reshape(-1, 3), dataset, distance)
    return idx.reshape(n, m)


def build_mosaic(nearest, img_paths, width, height, grid_w, grid_h):
    """Susun gambar mosaik dari matriks indeks tile. Tile dibuka sekali per indeks."""
    size = (max(int(grid_w), 1), max(int(grid_h), 1))
    resized = {}
    result = Image.new("RGB", (width, height))
    rows, cols = nearest.shape
    for row in range(rows):
        for col in range(cols):
            k = int(nearest[row, col])
            if k not in resized:
                with Image.open(img_paths[k]) as t:
                    resized[k] = t.convert("RGB").resize(size)
            result.paste(resized[k], (int(col * grid_w), int(row * grid_h)))
    return result
