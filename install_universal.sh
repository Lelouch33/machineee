#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
# Gonka Node — vLLM 0.14.0 + CUDA 13.0 + Gonka PoC Module (V1 Engine)
# UNIVERSAL: For H100, H200, A100 and other NVIDIA GPUs
# NOT COMPATIBLE with B300/Blackwell (use install_b300.sh)
# ═══════════════════════════════════════════════════════════════════════════
set -euo pipefail

VLLM_VERSION="0.14.0"
CUDA_VERSION="13.0"
SCRIPT_NAME="install_universal"

#################################
# COLORS
#################################
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

log_info()    { echo -e "${BLUE}[INFO]${NC} $(date '+%H:%M:%S') $1"; }
log_success() { echo -e "${GREEN}[✓]${NC} $(date '+%H:%M:%S') $1"; }
log_warning() { echo -e "${YELLOW}[⚠]${NC} $(date '+%H:%M:%S') $1"; }
log_error()   { echo -e "${RED}[✗]${NC} $(date '+%H:%M:%S') $1"; }
log_poc()     { echo -e "${CYAN}[PoC]${NC} $(date '+%H:%M:%S') $1"; }

#################################
# CHECK ROOT
#################################
if [[ $EUID -ne 0 ]]; then
    log_error "Run as root: sudo bash $0"
    exit 1
fi

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║     Gonka Node — vLLM ${VLLM_VERSION} + CUDA ${CUDA_VERSION}                      ║"
echo "║     UNIVERSAL: H100/H200/A100 (V1 Engine)                        ║"
echo "║     WITH: gonka_poc module                                       ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

#################################
# CONFIG
#################################
HF_CACHE="/root/.cache"
LOG_DIR="/var/log/gonka"
LOG_FILE="$LOG_DIR/uvicorn.log"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

#################################
# 1. BASE PACKAGES
#################################
log_info "Installing base packages..."
apt-get update
apt-get install -y --no-install-recommends \
    ca-certificates curl git jq tar wget \
    nginx tmux lsof \
    software-properties-common

#################################
# 2. CUDA TOOLKIT
#################################
if [ ! -d "/usr/local/cuda-13.0" ]; then
    log_warning "CUDA 13.0 toolkit not found. Installing..."
    if ! dpkg -l | grep -q cuda-keyring; then
        log_info "Adding NVIDIA repository..."
        wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb -O /tmp/cuda-keyring.deb
        dpkg -i /tmp/cuda-keyring.deb
        apt-get update
    fi
    log_info "Installing CUDA 13.0 toolkit (this may take a few minutes)..."
    apt-get install -y cuda-toolkit-13-0
    log_success "CUDA 13.0 toolkit installed"
fi

# Set CUDA_HOME
if [ -d "/usr/local/cuda-13.0" ]; then
    CUDA_HOME="/usr/local/cuda-13.0"
elif [ -d "/usr/local/cuda-12.6" ]; then
    CUDA_HOME="/usr/local/cuda-12.6"
    log_warning "Using CUDA 12.6"
elif [ -d "/usr/local/cuda" ]; then
    CUDA_HOME="/usr/local/cuda"
else
    log_error "CUDA not found"
    exit 1
fi
log_info "CUDA_HOME: $CUDA_HOME"

#################################
# 3. ULIMIT
#################################
log_info "Setting ulimits..."
ulimit -n 65536
ulimit -u 65536
cat >> /etc/security/limits.conf << 'LIMITS'
* soft nofile 65536
* hard nofile 65536
* soft nproc 65536
* hard nproc 65536
LIMITS

#################################
# 4. PYTHON 3.12 + UV
#################################
log_info "Installing Python 3.12..."
add-apt-repository -y ppa:deadsnakes/ppa 2>/dev/null || true
apt-get update
apt-get install -y python3.12 python3.12-venv python3.12-dev

log_info "Installing uv package manager..."
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
if command -v uv &> /dev/null; then
    log_success "uv installed: $(uv --version)"
else
    log_warning "uv failed, using pip"
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12 - --break-system-packages
fi

#################################
# 5. CHECK GPU
#################################
log_info "Checking GPU..."
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
GPU_COUNT=$(nvidia-smi -L | wc -l)
log_success "Found $GPU_COUNT GPUs"

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
log_info "GPU Model: $GPU_NAME"

# Block Blackwell GPUs (use install_b300.sh instead)
if [[ "$GPU_NAME" == *"B300"* ]] || [[ "$GPU_NAME" == *"Blackwell"* ]]; then
    log_error "Blackwell GPU detected (B300)! Use install_b300.sh instead."
    exit 1
fi

if [[ "$GPU_NAME" == *"H100"* ]] || [[ "$GPU_NAME" == *"H200"* ]]; then
    log_success "Detected Hopper GPU"
elif [[ "$GPU_NAME" == *"A100"* ]]; then
    log_success "Detected Ampere A100 GPU"
else
    log_info "GPU Model: $GPU_NAME - proceeding with universal config"
fi

DRIVER_VERSION_FULL=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
DRIVER_MAJOR=$(echo "$DRIVER_VERSION_FULL" | cut -d'.' -f1)
log_info "NVIDIA Driver: $DRIVER_VERSION_FULL"

MIN_DRIVER_MAJOR=535
if [ "$DRIVER_MAJOR" -lt "$MIN_DRIVER_MAJOR" ]; then
    log_error "Driver version $DRIVER_VERSION_FULL is too old!"
    exit 1
fi
log_success "Driver version OK"

#################################
# 6. ENVIRONMENT
#################################
export CUDA_HOME="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export HF_HOME="$HF_CACHE"
export TRANSFORMERS_CACHE="$HF_CACHE"

#################################
# 7. REMOVE CONFLICTING PACKAGES
#################################
log_info "Removing conflicting packages..."
apt-get remove -y python3-torch python3-triton 2>/dev/null || true
rm -rf /usr/lib/python3/dist-packages/torch* 2>/dev/null || true
rm -rf /usr/lib/python3/dist-packages/triton* 2>/dev/null || true

#################################
# 8. INSTALL PYTORCH + VLLM
#################################
log_info "Installing vLLM ${VLLM_VERSION} with CUDA ${CUDA_VERSION}..."
uv pip install --python python3.12 --system --break-system-packages --index-strategy unsafe-best-match \
    vllm==${VLLM_VERSION} \
    --extra-index-url https://wheels.vllm.ai/${VLLM_VERSION}/cu130 \
    --extra-index-url https://download.pytorch.org/whl/cu130

log_info "Installing dependencies..."
uv pip install --python python3.12 --system --break-system-packages --index-strategy unsafe-best-match \
    uvicorn fastapi starlette pydantic \
    "httpx[http2]" toml fire nvidia-ml-py \
    accelerate tiktoken transformers \
    openai aiohttp \
    grpcio grpcio-tools protobuf

apt-get remove -y python3-scipy 2>/dev/null || true
uv pip install --python python3.12 --system --break-system-packages --index-strategy unsafe-best-match scipy

#################################
# 9. FLASHINFER + TRITON
#################################
log_info "Installing FlashInfer 0.5.3..."
uv pip uninstall --python python3.12 --system --break-system-packages flashinfer-python 2>/dev/null || true
uv pip install --python python3.12 --system --break-system-packages --index-strategy unsafe-best-match \
    flashinfer-python==0.5.3 \
    --extra-index-url https://flashinfer.ai/whl/cu130/torch2.5/

apt-get remove -y python3-optree 2>/dev/null || true
uv pip install --python python3.12 --system --break-system-packages --index-strategy unsafe-best-match optree
uv pip install --python python3.12 --system --break-system-packages --index-strategy unsafe-best-match triton 2>/dev/null || true

rm -rf ~/.cache/flashinfer 2>/dev/null || true

#################################
# 10. VERIFY INSTALLATION
#################################
log_info "Verifying installation..."
echo "  PyTorch: $(python3.12 -c 'import torch; print(torch.__version__)')"
echo "  CUDA: $(python3.12 -c 'import torch; print(torch.version.cuda)')"
echo "  vLLM: $(python3.12 -c 'import vllm; print(vllm.__version__)')"
python3.12 -c "import flashinfer; print('  FlashInfer: installed')" 2>/dev/null || echo "  FlashInfer: not installed"
python3.12 -c "import triton; print(f'  Triton: {triton.__version__}')" 2>/dev/null || echo "  Triton: not installed"

#################################
# 11. INSTALL GONKA PoC MODULE
#################################
log_poc "Installing gonka_poc module (V1 Engine)..."

VLLM_PATH=$(python3.12 -c "import vllm; print(vllm.__path__[0])")
GONKA_POC_DEST="$VLLM_PATH/gonka_poc"

# Find gonka_poc directory
GONKA_POC_SOURCE=""
for CHECK_DIR in "$SCRIPT_DIR/gonka_poc" "./gonka_poc" "$(dirname "$SCRIPT_DIR")/gonka_poc"; do
    if [ -d "$CHECK_DIR" ] && [ -f "$CHECK_DIR/__init__.py" ] && [ -f "$CHECK_DIR/routes.py" ]; then
        GONKA_POC_SOURCE="$CHECK_DIR"
        break
    fi
done

if [ -n "$GONKA_POC_SOURCE" ]; then
    log_poc "Using local gonka_poc from: $GONKA_POC_SOURCE"
    rm -rf "$GONKA_POC_DEST"
    cp -r "$GONKA_POC_SOURCE" "$VLLM_PATH/"
    log_success "gonka_poc installed from local source"
else
    log_error "gonka_poc directory not found! Please ensure it's in the same directory as install script."
    exit 1
fi

# Verify installation
if python3.12 -c "from vllm.gonka_poc import PoCManagerV1" 2>/dev/null; then
    log_success "gonka_poc module installed successfully"
else
    log_error "gonka_poc module installation failed!"
    exit 1
fi

#################################
# 12. PATCH api_server.py
#################################
log_poc "Patching api_server.py for gonka_poc support..."

API_SERVER_PATH=$(python3.12 -c "import vllm.entrypoints.openai; import os; print(os.path.dirname(vllm.entrypoints.openai.__file__))")
API_SERVER_FILE="$API_SERVER_PATH/api_server.py"

if ! grep -q "from vllm.gonka_poc.routes import router as gonka_poc_router" "$API_SERVER_FILE"; then
    cp "$API_SERVER_FILE" "$API_SERVER_FILE.bak"
    LAST_IMPORT_LINE=$(grep -n "^from " "$API_SERVER_FILE" | tail -1 | cut -d: -f1)
    sed -i "${LAST_IMPORT_LINE}a from vllm.gonka_poc.routes import router as gonka_poc_router" "$API_SERVER_FILE"
    sed -i "/app.include_router(router)/a\    app.include_router(gonka_poc_router)" "$API_SERVER_FILE"
    log_success "api_server.py patched for gonka_poc"
else
    log_info "api_server.py already patched"
fi

#################################
# 13. NGINX CONFIG
#################################
log_info "Configuring nginx..."

cat > /etc/nginx/sites-available/gonka << 'NGINX'
log_format gonka '$host:$server_port [$time_local] "$request" '
                 '$status $body_bytes_sent "$http_user_agent"';

client_max_body_size 0;
proxy_connect_timeout 24h;
proxy_send_timeout 24h;
proxy_read_timeout 24h;
proxy_http_version 1.1;
proxy_set_header Connection "";

upstream api_8080 { server 127.0.0.1:8080; }
upstream vllm_5000 { server 127.0.0.1:5000; }

server {
    listen 8081;
    server_name _;
    access_log /var/log/nginx/gonka_access.log gonka;

    location /v3.0.8/ {
        proxy_pass http://api_8080/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location / {
        proxy_pass http://api_8080/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}

server {
    listen 5050;
    server_name _;
    access_log /var/log/nginx/gonka_access.log gonka;

    location /v3.0.8/ {
        proxy_pass http://vllm_5000/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location / {
        proxy_pass http://vllm_5000/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
NGINX

ln -sf /etc/nginx/sites-available/gonka /etc/nginx/sites-enabled/gonka
rm -f /etc/nginx/sites-enabled/default
nginx -t && nginx -s reload || systemctl restart nginx
log_success "nginx configured"

#################################
# 14. LOGS
#################################
mkdir -p "$LOG_DIR"
touch "$LOG_FILE"
chmod 644 "$LOG_FILE"
cat > /etc/logrotate.d/gonka << 'LR'
/var/log/gonka/*.log {
    daily rotate 2 missingok notifempty compress delaycompress copytruncate
}
LR

#################################
# 15. DOWNLOAD GONKA APP
#################################
log_info "Downloading Gonka app..."
apt-get install -y skopeo

GONKA_IMAGE="ghcr.io/product-science/mlnode@sha256:59ef2068648f2a72330151d90a6aa4f1106c55a7fa2ad031d5f10fa282f97da2"
WORKDIR="$HOME/gonka"
mkdir -p "$WORKDIR"
cd "$WORKDIR"

if [ ! -d mlnode_oci ]; then
    log_info "Pulling image..."
    skopeo copy "docker://${GONKA_IMAGE}" oci:mlnode_oci:latest
fi

APP="$WORKDIR/bundle/rootfs/app"

if [ -d "$APP/packages" ]; then
    log_success "Gonka app already extracted at $APP"
else
    log_info "Extracting app..."
    mkdir -p bundle/rootfs
    find mlnode_oci/blobs/sha256 -maxdepth 1 -type f -print0 \
        | xargs -0 -I{} tar -xf {} -C bundle/rootfs 2>/dev/null || true
    ls -la "$APP/packages" || { log_error "App extraction failed!"; exit 1; }
    log_success "Gonka app extracted to $APP"
fi

#################################
# 16. PATCH RUNNER.PY FOR V1
#################################
log_poc "Patching runner.py for V1 engine..."

RUNNER_FILE="$APP/packages/api/src/api/inference/vllm/runner.py"

if [ -f "$RUNNER_FILE" ]; then
    # Backup
    cp "$RUNNER_FILE" "$RUNNER_FILE.bak"
    
    # Add comma after VLLM_HOST if missing
    sed -i 's/"--host", self.VLLM_HOST$/"--host", self.VLLM_HOST,/' "$RUNNER_FILE"
    
    # Add --enforce-eager if not present
    if ! grep -q '"--enforce-eager"' "$RUNNER_FILE"; then
        sed -i '/"--host", self.VLLM_HOST,/a\                "--enforce-eager",' "$RUNNER_FILE"
    fi
    
    # Change V1 mode
    sed -i 's/env\["VLLM_USE_V1"\] = "0"/env["VLLM_USE_V1"] = "1"/' "$RUNNER_FILE"
    
    # Add env vars if not present
    if ! grep -q 'VLLM_USE_CUDA_GRAPHS' "$RUNNER_FILE"; then
        sed -i '/env\["VLLM_USE_V1"\] = "1"/a\            env["VLLM_USE_CUDA_GRAPHS"] = "0"' "$RUNNER_FILE"
    fi
    if ! grep -q 'VLLM_ALLOW_INSECURE_SERIALIZATION' "$RUNNER_FILE"; then
        sed -i '/env\["VLLM_USE_CUDA_GRAPHS"\] = "0"/a\            env["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"' "$RUNNER_FILE"
    fi
    
    # Verify syntax
    if python3.12 -m py_compile "$RUNNER_FILE" 2>/dev/null; then
        log_success "runner.py patched for V1 engine"
    else
        log_error "runner.py patch failed! Restoring backup..."
        cp "$RUNNER_FILE.bak" "$RUNNER_FILE"
        exit 1
    fi
else
    log_warning "runner.py not found, skipping patch"
fi

#################################
# 17. PYTHONPATH
#################################
export PYTHONPATH="$APP/packages/api/src:$APP/packages/train/src:$APP/packages/common/src:$APP/packages/pow/src"

#################################
# 18. START IN TMUX
#################################
TMUX_SESSION="gonka"
log_info "Starting Gonka in tmux session: $TMUX_SESSION"

if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
    log_warning "Stopping existing tmux session..."
    tmux kill-session -t "$TMUX_SESSION"
    sleep 2
fi

log_info "Cleaning up orphaned processes..."
pkill -9 -f "uvicorn api.app:app" 2>/dev/null || true
pkill -9 -f "python.*pow" 2>/dev/null || true
pkill -9 -f "python.*vllm" 2>/dev/null || true
pkill -9 -f "vllm.entrypoints" 2>/dev/null || true
sleep 3

# Kill any remaining processes holding GPU memory
log_info "Checking for processes on GPU..."
GPU_PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sort -u)
if [ -n "$GPU_PIDS" ]; then
    log_warning "Found processes on GPU, killing: $GPU_PIDS"
    echo "$GPU_PIDS" | xargs -I{} kill -9 {} 2>/dev/null || true
    sleep 5
    # Second check
    GPU_PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sort -u)
    if [ -n "$GPU_PIDS" ]; then
        log_error "Still have processes on GPU after kill: $GPU_PIDS"
        log_error "Please manually stop these processes and re-run the script."
        exit 1
    fi
    log_success "All GPU processes terminated"
else
    log_success "No processes on GPU"
fi

GPU_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
GPU_TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
log_info "GPU memory: ${GPU_USED}MB used / ${GPU_TOTAL}MB total"

FLASHINFER_AVAILABLE=$(python3.12 -c "import flashinfer" 2>/dev/null && echo "1" || echo "0")
ACTUAL_VLLM=$(python3.12 -c 'import vllm; print(vllm.__version__)')
GONKA_POC_CHECK=$(python3.12 -c "from vllm.gonka_poc import PoCManagerV1; print('OK')" 2>/dev/null || echo "FAIL")

tmux new-session -d -s "$TMUX_SESSION" "
cd '$APP'
export PYTHONPATH='$PYTHONPATH'
export HF_HOME='$HF_CACHE'
export TRANSFORMERS_CACHE='$HF_CACHE'
export CUDA_HOME='$CUDA_HOME'
export PATH='$CUDA_HOME/bin:/usr/local/bin:/usr/bin:/bin'
export LD_LIBRARY_PATH='$CUDA_HOME/lib64:\${LD_LIBRARY_PATH:-}'
[ '$FLASHINFER_AVAILABLE' = '1' ] && export VLLM_ATTENTION_BACKEND=FLASHINFER
echo '══════════════════════════════════════════════════════════════'
echo 'Gonka Node — Universal (V1 Engine)'
echo \"vLLM: $ACTUAL_VLLM | CUDA: ${CUDA_VERSION}\"
echo \"gonka_poc: $GONKA_POC_CHECK\"
echo '══════════════════════════════════════════════════════════════'
/usr/bin/python3.12 -m uvicorn api.app:app --host 127.0.0.1 --port 8080
"

tmux pipe-pane -t "$TMUX_SESSION" "cat >> $LOG_FILE"

#################################
# 19. HEALTH CHECK
#################################
log_info "Waiting for service to start..."
sleep 10

for i in {1..60}; do
    if curl -s -f -m 5 "http://127.0.0.1:8080/health" > /dev/null 2>&1; then
        log_success "Service is healthy!"
        break
    fi
    echo -n "."
    sleep 5
done
echo ""

#################################
# 20. FINAL STATUS
#################################
echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║     INSTALLATION COMPLETE — UNIVERSAL (V1 Engine)               ║"
echo "╠══════════════════════════════════════════════════════════════════╣"
echo "║  vLLM: ${VLLM_VERSION} (V1 Engine + enforce-eager)                    ║"
echo "║  gonka_poc: $GONKA_POC_CHECK                                            ║"
echo "╠══════════════════════════════════════════════════════════════════╣"
echo "║  PoC Endpoints: /api/v1/pow/*                                    ║"
echo "║  Ports: API=8081, Inference=5050                                 ║"
echo "║  Tmux: tmux attach -t $TMUX_SESSION                                 ║"
echo "║  Logs: tail -f $LOG_FILE                         ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

if curl -s "http://127.0.0.1:8080/health" > /dev/null 2>&1; then
    log_success "Gonka node is running!"
else
    log_error "Service may not be running. Check: tail -f $LOG_FILE"
fi
