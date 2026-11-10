"""app.py - REST API Photomosaic (KNN) dengan FastAPI."""
import io
import os
import pickle

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

from mosaic_core import (DISTANCES, FOLDER_CATEGORY, build_mosaic,
                         get_avg_rgb_per_grid, knn_grid, knn_nearest)

# Konfigurasi dari environment variable (diisi Docker / Kubernetes)
APP_VERSION = os.getenv("APP_VERSION", "v1")
APP_ENV = os.getenv("APP_ENV", "development")
MODEL_PATH = os.getenv("MODEL_PATH", "model.pkl")
DEFAULT_DISTANCE = os.getenv("DEFAULT_DISTANCE", "euclidean")
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "5"))
MAX_GRID = int(os.getenv("MAX_GRID", "128"))
MAX_SIDE = int(os.getenv("MAX_SIDE", "1024"))  # sisi terpanjang gambar (px)
POD = os.getenv("HOSTNAME", "local")

# Indeks KNN dimuat SEKALI saat start
with open(MODEL_PATH, "rb") as f:
    INDEX = pickle.load(f)

app = FastAPI(title="Photomosaic KNN Service", version=APP_VERSION)


class PredictInput(BaseModel):
    category: str = Field(..., examples=["mountain"])
    rgb: list[float] = Field(..., min_length=3, max_length=3, examples=[[120, 130, 140]])
    distance: str = Field(DEFAULT_DISTANCE, examples=["euclidean"])


def check_category(category: str) -> str:
    category = category.lower()
    if category not in INDEX:
        raise HTTPException(400, f"category tidak valid. Pilihan: {list(FOLDER_CATEGORY)}")
    return category


def check_distance(distance: str) -> str:
    distance = distance.lower()
    if distance not in DISTANCES:
        raise HTTPException(400, f"distance tidak valid. Pilihan: {DISTANCES}")
    return distance


@app.get("/")
def root():
    return {"service": "mosaic-api", "version": APP_VERSION, "environment": APP_ENV,
            "algorithm": "KNN (k=1)", "docs": "/docs", "pod": POD}


@app.get("/health")
def health():
    """Dipakai Kubernetes untuk readiness/liveness probe."""
    return {"status": "ok"}


@app.get("/categories")
def categories():
    return list(FOLDER_CATEGORY)


@app.get("/distances")
def distances():
    return DISTANCES


@app.post("/predict")
def predict(data: PredictInput):
    """KNN untuk satu warna: tile mana yang paling dekat?"""
    category = check_category(data.category)
    distance = check_distance(data.distance)
    if not all(0 <= v <= 255 for v in data.rgb):
        raise HTTPException(422, "nilai rgb harus 0-255")
    tiles = INDEX[category]
    import numpy as np
    idx, dist = knn_nearest(np.array([data.rgb], dtype=float), tiles["avg_rgb"], distance)
    i = int(idx[0])
    return {"category": category, "distance_metric": distance,
            "tile": tiles["paths"][i],
            "tile_rgb": [round(float(v), 2) for v in tiles["avg_rgb"][i]],
            "distance": round(float(dist[0]), 4),
            "version": APP_VERSION, "pod": POD}


@app.post("/mosaic")
def mosaic(image: UploadFile = File(...),
           category: str = Form(...),
           distance: str = Form(DEFAULT_DISTANCE),
           grid: int = Form(64)):
    """Upload gambar, balas JPEG photomosaic."""
    category = check_category(category)
    distance = check_distance(distance)
    if not 8 <= grid <= MAX_GRID:
        raise HTTPException(422, f"grid harus 8-{MAX_GRID}")

    raw = image.file.read(MAX_UPLOAD_MB * 1024 * 1024 + 1)
    if len(raw) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"file maksimal {MAX_UPLOAD_MB} MB")
    try:
        img = Image.open(io.BytesIO(raw))
        img.load()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, "file bukan gambar yang valid")

    img = img.convert("RGB")
    img.thumbnail((MAX_SIDE, MAX_SIDE))  # cegah OOM untuk gambar besar
    avg, width, height, grid_w, grid_h = get_avg_rgb_per_grid(img, grid, grid)
    tiles = INDEX[category]
    nearest = knn_grid(avg, tiles["avg_rgb"], distance)
    result = build_mosaic(nearest, tiles["paths"], width, height, grid_w, grid_h)

    buf = io.BytesIO()
    result.save(buf, format="JPEG", quality=90)
    return Response(content=buf.getvalue(), media_type="image/jpeg",
                    headers={"X-Pod": POD, "X-Version": APP_VERSION,
                             "X-Category": category, "X-Distance": distance})
