#!/usr/bin/env bash
# smoke_test.sh - Tes otomatis Mosaic API. Cetak PASS/FAIL per tes.
#
# Pemakaian:
#   ./smoke_test.sh                       # ke http://localhost:8000
#   ./smoke_test.sh $URL                  # mis. URL=$(minikube service mosaic-api --url)
#   EXPECT_PODS=2 ./smoke_test.sh $URL    # wajib melihat >= 2 nama pod berbeda
#   SAMPLE=foto.jpg ./smoke_test.sh $URL  # gambar contoh untuk /mosaic (default sample.png)

URL="${1:-http://localhost:8000}"
URL="${URL%/}"
SAMPLE="${SAMPLE:-sample.png}"
EXPECT_PODS="${EXPECT_PODS:-1}"
TMP="$(mktemp -d)"
PASS=0
FAIL=0

ok()   { echo "PASS  $1"; PASS=$((PASS + 1)); }
bad()  { echo "FAIL  $1  ($2)"; FAIL=$((FAIL + 1)); }

# expect_code NAMA KODE_DIHARAPKAN [argumen curl ...]  -> body ada di $TMP/body
expect_code() {
  local name="$1" want="$2"; shift 2
  local got
  got=$(curl -s -o "$TMP/body" -w "%{http_code}" "$@")
  if [ "$got" = "$want" ]; then ok "$name"; else bad "$name" "kode $got, harusnya $want"; fi
}

body_has() {  # body_has NAMA TEKS
  if grep -q "$2" "$TMP/body"; then ok "$1"; else bad "$1" "respons tidak memuat '$2'"; fi
}

JSON='Content-Type: application/json'

echo "== Mosaic API smoke test: $URL =="

echo "-- Dasar"
expect_code "GET /health"        200 "$URL/health"
expect_code "GET /"              200 "$URL/";            body_has "GET / memuat mosaic-api" "mosaic-api"
expect_code "GET /categories"    200 "$URL/categories";  body_has "categories memuat mountain" "mountain"
expect_code "GET /distances"     200 "$URL/distances";   body_has "distances memuat manhattan" "manhattan"

echo "-- /predict (KNN satu warna)"
for d in euclidean minkowski manhattan; do
  expect_code "predict $d" 200 -X POST "$URL/predict" -H "$JSON" \
    -d "{\"category\":\"mountain\",\"rgb\":[120,130,140],\"distance\":\"$d\"}"
  body_has "predict $d memuat tile" '"tile"'
done
expect_code "predict kategori salah" 400 -X POST "$URL/predict" -H "$JSON" -d '{"category":"xyz","rgb":[1,2,3]}'
expect_code "predict distance salah" 400 -X POST "$URL/predict" -H "$JSON" -d '{"category":"cloud","rgb":[1,2,3],"distance":"cosine"}'
expect_code "predict rgb hanya 2 angka" 422 -X POST "$URL/predict" -H "$JSON" -d '{"category":"cloud","rgb":[1,2]}'
expect_code "predict rgb di luar 0-255" 422 -X POST "$URL/predict" -H "$JSON" -d '{"category":"cloud","rgb":[1,2,300]}'

echo "-- /mosaic (upload gambar)"
if [ -f "$SAMPLE" ]; then
  for d in euclidean minkowski manhattan; do
    ctype=$(curl -s -o "$TMP/out_$d.jpg" -w "%{http_code} %{content_type}" -X POST "$URL/mosaic" \
      -F "image=@$SAMPLE" -F category=mountain -F distance=$d -F grid=32)
    if [ "$ctype" = "200 image/jpeg" ] && [ -s "$TMP/out_$d.jpg" ]; then
      ok "mosaic $d -> JPEG ($(wc -c < "$TMP/out_$d.jpg") byte)"
    else
      bad "mosaic $d" "dapat '$ctype'"
    fi
  done
  expect_code "mosaic kategori salah" 400 -X POST "$URL/mosaic" -F "image=@$SAMPLE" -F category=zzz
  expect_code "mosaic grid 999"       422 -X POST "$URL/mosaic" -F "image=@$SAMPLE" -F category=cloud -F grid=999
else
  echo "SKIP  tes /mosaic: file $SAMPLE tidak ditemukan (set SAMPLE=...)"
fi
echo "bukan gambar" > "$TMP/fake.txt"
expect_code "mosaic file bukan gambar" 400 -X POST "$URL/mosaic" -F "image=@$TMP/fake.txt" -F category=cloud

echo "-- Load balancing (nama pod)"
for i in $(seq 1 20); do
  curl -s -X POST "$URL/predict" -H "$JSON" -d '{"category":"cloud","rgb":[200,200,210]}' \
    | grep -o '"pod":"[^"]*"'
done | sort | uniq -c > "$TMP/pods"
PODS=$(wc -l < "$TMP/pods")
cat "$TMP/pods"
if [ "$PODS" -ge "$EXPECT_PODS" ]; then ok "terlihat $PODS pod berbeda (minimal $EXPECT_PODS)"
else bad "load balancing" "hanya $PODS pod, harusnya >= $EXPECT_PODS"; fi

rm -rf "$TMP"
echo "== Ringkasan: $PASS PASS, $FAIL FAIL =="
[ "$FAIL" -eq 0 ]
