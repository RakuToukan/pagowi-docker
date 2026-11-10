# 1. Image dasar Python ringan
FROM python:3.12-slim

# 2. Log langsung tampil, tanpa file .pyc
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# 3. Install library dulu (memanfaatkan cache layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Salin kode dan dataset tile, lalu bangun indeks KNN DI DALAM image
COPY mosaic_core.py build_index.py app.py ./
COPY assets/ assets/
RUN python build_index.py

# 5. Versi aplikasi bisa diganti saat build: --build-arg APP_VERSION=v2
ARG APP_VERSION=v1
ENV APP_VERSION=${APP_VERSION}

# 6. Jalankan sebagai user biasa, bukan root
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
