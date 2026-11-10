#!/usr/bin/env bash
# deploy.sh - Deploy Mosaic API ke minikube (Cloud Shell) dan kumpulkan bukti.
#
# Pemakaian (dari folder mosaic-ai-service):
#   chmod +x deploy.sh smoke_test.sh && ./deploy.sh
#
# Langkah: build image -> tes container -> minikube start -> build di minikube
#          -> apply ConfigMap/Deployment/Service -> smoke test -> simpan bukti.
# Hasil lengkap disimpan di bukti_deployment.txt (untuk laporan).

set -o pipefail
cd "$(dirname "$0")" || exit 1

run() {
  set -e
  step() { echo; echo "=== $* ==="; }

  step "1. Build image mosaic-api:v1 (Docker Cloud Shell)"
  docker build -t mosaic-api:v1 .
  docker images mosaic-api

  step "2. Tes container (docker run)"
  docker rm -f mosaic-test >/dev/null 2>&1 || true
  docker run -d --name mosaic-test -p 8000:8000 mosaic-api:v1
  for _ in $(seq 1 30); do curl -s localhost:8000/health >/dev/null && break; sleep 1; done
  ./smoke_test.sh http://localhost:8000
  docker rm -f mosaic-test

  step "3. Nyalakan minikube"
  minikube status >/dev/null 2>&1 || minikube start || minikube start --memory=2200mb
  minikube addons enable metrics-server
  kubectl get nodes

  step "4. Build image ke dalam minikube"
  eval "$(minikube docker-env)"
  docker build -t mosaic-api:v1 .
  docker images | grep mosaic-api

  step "5. Deploy: ConfigMap, Deployment, Service"
  kubectl apply -f k8s/configmap.yaml -f k8s/deployment.yaml -f k8s/service.yaml
  kubectl rollout status deployment/mosaic-api --timeout=240s

  step "6. Bukti deployment"
  kubectl get deployment,replicaset,pod,service,configmap -o wide
  kubectl get endpoints mosaic-api

  step "7. Smoke test lewat Service (harus terlihat 2 pod berbeda)"
  URL=$(minikube service mosaic-api --url | head -1)
  echo "URL Service: $URL"
  EXPECT_PODS=2 ./smoke_test.sh "$URL"

  echo; echo "=== SELESAI: deployment berhasil ==="
}

run 2>&1 | tee bukti_deployment.txt
status=${PIPESTATUS[0]}
echo
if [ "$status" -eq 0 ]; then echo "OK. Bukti tersimpan di bukti_deployment.txt"
else echo "GAGAL (kode $status). Lihat pesan error di atas / bukti_deployment.txt"; fi
exit "$status"
