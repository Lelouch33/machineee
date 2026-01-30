#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
# Gonka Node — Auto-Installer
# Detects GPU type and runs appropriate installation script
# ═══════════════════════════════════════════════════════════════════════════
set -euo pipefail

#################################
# COLORS
#################################
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

log_info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[✓]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[⚠]${NC} $1"; }
log_error()   { echo -e "${RED}[✗]${NC} $1"; }

#################################
# CHECK ROOT
#################################
if [[ $EUID -ne 0 ]]; then
    log_error "Run as root: sudo bash $0"
    exit 1
fi

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║         Gonka Node — Auto-Installer v2.0 (V1 Engine)            ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

#################################
# DETECT GPU
#################################
log_info "Detecting GPU..."

if ! command -v nvidia-smi &> /dev/null; then
    log_error "nvidia-smi not found! NVIDIA driver required."
    exit 1
fi

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
log_info "GPU Model: ${CYAN}$GPU_NAME${NC}"

# Detect compute capability
GPU_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
log_info "Compute Capability: ${CYAN}$GPU_ARCH${NC}"

#################################
# SELECT INSTALL SCRIPT
#################################
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_SCRIPT=""

# Check for Blackwell B300/B200
if [[ "$GPU_NAME" == *"B300"* ]] || [[ "$GPU_NAME" == *"B200"* ]] || \
   [[ "$GPU_ARCH" == "10.0"* ]] || [[ "$GPU_ARCH" == "10.3"* ]]; then
    log_success "Detected: ${CYAN}Blackwell B300/B200${NC}"
    INSTALL_SCRIPT="$SCRIPT_DIR/install_b300.sh"

    if [ ! -f "$INSTALL_SCRIPT" ]; then
        log_error "install_b300.sh not found in $SCRIPT_DIR"
        exit 1
    fi

    log_info "Will run: install_b300.sh"

# Check for Hopper H100/H200
elif [[ "$GPU_NAME" == *"H100"* ]] || [[ "$GPU_NAME" == *"H200"* ]] || \
     [[ "$GPU_ARCH" == "9.0"* ]]; then
    log_success "Detected: ${CYAN}Hopper H100/H200${NC}"
    INSTALL_SCRIPT="$SCRIPT_DIR/install_universal.sh"

    if [ ! -f "$INSTALL_SCRIPT" ]; then
        log_error "install_universal.sh not found in $SCRIPT_DIR"
        exit 1
    fi

    log_info "Will run: install_universal.sh"

# Check for Ampere A100
elif [[ "$GPU_NAME" == *"A100"* ]] || [[ "$GPU_ARCH" == "8.0"* ]]; then
    log_success "Detected: ${CYAN}Ampere A100${NC}"
    INSTALL_SCRIPT="$SCRIPT_DIR/install_universal.sh"

    if [ ! -f "$INSTALL_SCRIPT" ]; then
        log_error "install_universal.sh not found in $SCRIPT_DIR"
        exit 1
    fi

    log_info "Will run: install_universal.sh"

# Default to universal
else
    log_warning "Unknown GPU: $GPU_NAME"
    log_info "Defaulting to universal script..."
    INSTALL_SCRIPT="$SCRIPT_DIR/install_universal.sh"

    if [ ! -f "$INSTALL_SCRIPT" ]; then
        log_error "install_universal.sh not found in $SCRIPT_DIR"
        exit 1
    fi
fi

#################################
# CHECK GONKA_POC
#################################
if [ ! -d "$SCRIPT_DIR/gonka_poc" ]; then
    log_error "gonka_poc directory not found in $SCRIPT_DIR"
    log_error "Please ensure the gonka_poc module is present."
    exit 1
fi
log_success "gonka_poc module found"

#################################
# CONFIRM
#################################
echo ""
log_info "Installation script: ${CYAN}$(basename "$INSTALL_SCRIPT")${NC}"
echo ""
read -p "Continue? (y/N): " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    log_warning "Aborted by user"
    exit 0
fi

#################################
# RUN INSTALL SCRIPT
#################################
echo ""
log_info "Starting installation..."
echo ""

chmod +x "$INSTALL_SCRIPT"
exec bash "$INSTALL_SCRIPT"
