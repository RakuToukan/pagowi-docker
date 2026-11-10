# Photomosaic KNN Service

AI service pembuat **photomosaic**: gambar yang diunggah dibagi menjadi grid, lalu tiap kotak diganti dengan foto dari dataset yang warnanya paling mirip. Algoritmanya **KNN (k=1)** dengan tiga pilihan jarak: Euclidean, Minkowski (p=3), dan Manhattan.

Layanan dibungkus dengan **FastAPI**, dikemas dengan **Docker**, dan dijalankan di **Kubernetes (minikube)** pada **Google Cloud Shell**. Proyek ini mengikuti panduan praktikum "Membangun AI Service dengan Docker dan Kubernetes di Google Cloud Shell". Data dan logika mosaik berasal dari proyek `uas_kdka`.

## Cara kerja KNN di sini

1. Gambar dibagi menjadi grid `grid x grid` kotak (default 64 x 64).
2. Tiap kotak dihitung warna rata-ratanya (R, G, B).
3. Dicari tile dataset yang jarak warnanya paling kecil (tetangga terdekat, k=1).
4. Tile tersebut ditempel di posisi kotak, dan hasilnya dikembalikan sebagai JPEG.

| Jarak | Rumus untuk dua warna `a` dan `b` |
|---|---|
| `euclidean` | akar dari jumlah kuadrat selisih R, G, B |
| `minkowski` | pangkat tiga akar dari jumlah selisih mutlak pangkat 3 (p=3) |
| `manhattan` | jumlah selisih mutlak R, G, B |

KNN tidak "belajar". Tahap `build_index.py` hanya menghitung warna rata-rata tiap tile dan menyimpannya ke `model.pkl`, supaya API tidak membaca ribuan gambar setiap kali dijalankan.

## Struktur proyek

```
mosaic-ai-service/
├── app.py              # REST API (FastAPI)
├── mosaic_core.py      # fungsi inti: rata-rata warna, KNN (NumPy), penyusun mosaik
├── build_index.py      # membuat model.pkl (warna rata-rata tiap tile)
├── requirements.txt    # library Python (versi dikunci)
├── Dockerfile
├── .dockerignore
├── assets/             # dataset tile: Building, Cloud, Forest, Mountain (~1.600 foto)
├── k8s/
│   ├── configmap.yaml  # konfigurasi (APP_ENV, DEFAULT_DISTANCE)
│   ├── deployment.yaml # 2 pod, rolling update, probe, resource limit
│   ├── service.yaml    # NodePort 30080
│   └── hpa.yaml        # autoscaling 2-5 pod berdasarkan CPU
├── smoke_test.sh       # tes otomatis API (PASS/FAIL)
├── compare_knn.py      # bukti KNN NumPy sama dengan KNN loop asli
├── main_original.py    # kode KNN asli (untuk pembanding)
└── sample.png          # gambar contoh untuk uji
```

## Endpoint API

| Method | Path | Fungsi |
|---|---|---|
| GET | `/` | info service: versi, environment, nama pod |
| GET | `/health` | health check (dipakai probe Kubernetes) |
| GET | `/categories` | daftar kategori tile |
| GET | `/distances` | daftar rumus jarak |
| POST | `/predict` | KNN untuk satu warna: tile terdekat dan jaraknya |
| POST | `/mosaic` | upload gambar, balasan berupa JPEG mosaik |
| GET | `/docs` | Swagger UI |

Kategori: `building`, `cloud`, `forest`, `mountain`. Jarak: `euclidean`, `minkowski`, `manhattan`.

Contoh `/predict`:

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"category":"mountain","rgb":[120,130,140],"distance":"manhattan"}'
```

Contoh `/mosaic` (field: `image`, `category`, `distance`, `grid` 8-128):

```bash
curl -X POST http://localhost:8000/mosaic \
  -F image=@sample.png -F category=mountain -F distance=euclidean -F grid=64 \
  -o hasil.jpg
```

Header respons `/mosaic` memuat `X-Pod` dan `X-Version`, jadi terlihat pod mana yang melayani. Batas: file maksimal 5 MB, sisi gambar diperkecil ke 1024 px.

Konfigurasi lewat environment variable: `APP_VERSION`, `APP_ENV`, `DEFAULT_DISTANCE`, `MAX_UPLOAD_MB`, `MAX_GRID`, `MAX_SIDE`, `MODEL_PATH`.

## Prasyarat

- Akun Google (Gmail pribadi lebih aman; akun kampus kadang membatasi Cloud Shell).
- Cukup buka [shell.cloud.google.com](https://shell.cloud.google.com). Python, Docker, kubectl, dan minikube sudah tersedia. Tidak perlu kartu kredit.

Cek alat di Cloud Shell:

```bash
python3 --version && docker --version && kubectl version --client && minikube version
```

## 1. Ambil kode

```bash
git clone https://github.com/<username>/<repo>.git
cd <repo>
```

Cek dataset lengkap (Building, Cloud, Mountain masing-masing 400 file, Forest 401):

```bash
for d in assets/*; do echo "$d: $(ls "$d" | wc -l) file"; done
```

## 2. Jalankan tanpa Docker

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python build_index.py
uvicorn app:app --host 0.0.0.0 --port 8000
```

Buka tab terminal baru (ikon `+`) untuk mengetes:

```bash
chmod +x smoke_test.sh && ./smoke_test.sh
```

Hasil yang benar: `24 PASS, 0 FAIL`. Untuk mencoba lewat browser, klik **Web Preview**, pilih **Change port**, isi `8000`, lalu tambahkan `/docs` di URL. Tekan `Ctrl+C` untuk menghentikan server.

## 3. Jalankan dengan Docker

```bash
docker build -t mosaic-api:v1 .
docker run -d --name mosaic-test -p 8000:8000 mosaic-api:v1
curl http://localhost:8000/
```

Nilai `pod` berisi ID container. Bersihkan setelah selesai:

```bash
docker stop mosaic-test && docker rm mosaic-test
```

## 4. Jalankan di Kubernetes (minikube)

```bash
minikube start                       # jika gagal memori: minikube start --memory=2200mb
minikube addons enable metrics-server
eval $(minikube docker-env)          # arahkan docker ke daemon milik minikube
docker build -t mosaic-api:v1 .
kubectl apply -f k8s/configmap.yaml -f k8s/deployment.yaml -f k8s/service.yaml
kubectl get pods -w                  # tunggu 2 pod 1/1 Running, lalu Ctrl+C
```

Uji lewat Service:

```bash
URL=$(minikube service mosaic-api --url)
EXPECT_PODS=2 ./smoke_test.sh $URL
```

`EXPECT_PODS=2` mensyaratkan terlihat dua nama pod berbeda, sebagai bukti load balancing.

Catatan: `eval $(minikube docker-env)` hanya berlaku di tab terminal tempat perintah dijalankan. Di tab baru, jalankan lagi.

## 5. Eksperimen Kubernetes

**Self-healing dan skala manual**

```bash
kubectl delete pod <nama-salah-satu-pod>   # pod pengganti muncul otomatis
kubectl scale deployment mosaic-api --replicas=4
kubectl scale deployment mosaic-api --replicas=2
```

**Ganti konfigurasi tanpa build ulang**: ubah `APP_ENV` di `k8s/configmap.yaml`, lalu:

```bash
kubectl apply -f k8s/configmap.yaml
kubectl rollout restart deployment mosaic-api
curl $URL/
```

**Rolling update ke v2 tanpa downtime**

```bash
docker build --build-arg APP_VERSION=v2 -t mosaic-api:v2 .
```

Di tab pertama, pantau (semua baris harus `200`, versi berpindah dari v1 ke v2):

```bash
URL=$(minikube service mosaic-api --url)
while true; do
  curl -s -o /dev/null -w "%{http_code} " -X POST $URL/predict -H "Content-Type: application/json" \
    -d '{"category":"cloud","rgb":[200,200,210]}'
  curl -s $URL/ | grep -o '"version":"[^"]*"'
  sleep 0.5
done
```

Di tab kedua:

```bash
kubectl set image deployment/mosaic-api api=mosaic-api:v2
kubectl rollout status deployment mosaic-api
kubectl rollout undo deployment mosaic-api     # rollback ke v1
```

**Autoscaling (HPA)**

```bash
kubectl apply -f k8s/hpa.yaml
kubectl get hpa mosaic-api -w                  # tunggu TARGETS berisi angka, bukan <unknown>
```

Di tab kedua, bangkitkan beban dari dalam cluster:

```bash
kubectl run load-generator --image=busybox:1.36 --restart=Never -- /bin/sh -c \
  'while true; do wget -q -O- --header="Content-Type: application/json" --post-data="{\"category\":\"cloud\",\"rgb\":[200,200,210]}" http://mosaic-api/predict >/dev/null; done'
```

Dalam 1-3 menit, kolom REPLICAS naik hingga maksimal 5. Hentikan beban dengan `kubectl delete pod load-generator`; HPA menunggu sekitar 5 menit sebelum menurunkan jumlah pod.

## 6. Bersihkan

```bash
kubectl delete -f k8s/ --ignore-not-found
kubectl delete pod load-generator --ignore-not-found
minikube stop        # simpan cluster; minikube delete untuk menghapus total
```

## Troubleshooting

| Gejala | Penyebab umum | Solusi |
|---|---|---|
| `ErrImagePull` / `ImagePullBackOff` | image dibuild di Docker Cloud Shell, bukan di minikube | `eval $(minikube docker-env)`, build ulang |
| `CrashLoopBackOff` | error saat start | `kubectl logs <pod>` dan `kubectl logs <pod> --previous` |
| Pod `Running` tapi `READY 0/1` | readiness probe gagal | `kubectl describe pod <pod>`, lihat bagian Events |
| HPA menampilkan `<unknown>` | metrics-server belum siap | `minikube addons enable metrics-server`, tunggu 1-2 menit |
| `OOMKilled` | memori melewati limit 768Mi | kecilkan `grid` atau ukuran gambar |
| `minikube start` gagal | memori atau disk Cloud Shell | `minikube delete`, `minikube start --memory=2200mb`, `docker image prune -a` |
| Cloud Shell lama tidak dipakai | VM direset, minikube berhenti | `minikube start`, lalu ulangi bagian 4 |

Perintah diagnosis: `kubectl get pods`, `kubectl describe pod <nama>`, `kubectl logs <nama>`, `kubectl get events --sort-by=.lastTimestamp`.

## Catatan teknis

- KNN ditulis ulang dengan NumPy (broadcasting). `compare_knn.py` membuktikan hasilnya identik dengan versi loop asli dan jauh lebih cepat.
- Model dan indeks dibuat saat build image (`RUN python build_index.py`), jadi tidak ada ketergantungan file di luar image.
- `/mosaic` sinkron (langsung membalas JPEG), jadi aman dengan banyak pod tanpa penyimpanan status bersama.
- Container berjalan sebagai user biasa (bukan root).
