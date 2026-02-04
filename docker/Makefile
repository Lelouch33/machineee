# ═══════════════════════════════════════════════════════════════════════════
# Gonka MLNode Docker Build
# ═══════════════════════════════════════════════════════════════════════════

VERSION := 0.14.0
REGISTRY := ghcr.io/lelouch33
IMAGE_NAME := gonka-mlnode

# Build args
CUDA_VERSION := 13.0.0
VLLM_VERSION := 0.14.0

# ═══════════════════════════════════════════════════════════════════════════
# Build targets
# ═══════════════════════════════════════════════════════════════════════════

.PHONY: build push clean

# Build for H100/H200/A100 (TRITON_ATTN)
build:
	docker build \
		-f Dockerfile.universal \
		--build-arg CUDA_VERSION=$(CUDA_VERSION) \
		--build-arg VLLM_VERSION=$(VLLM_VERSION) \
		-t $(REGISTRY)/$(IMAGE_NAME):v$(VERSION) \
		-t $(REGISTRY)/$(IMAGE_NAME):latest \
		..

# ═══════════════════════════════════════════════════════════════════════════
# Push targets
# ═══════════════════════════════════════════════════════════════════════════

push:
	docker push $(REGISTRY)/$(IMAGE_NAME):v$(VERSION)
	docker push $(REGISTRY)/$(IMAGE_NAME):latest

# ═══════════════════════════════════════════════════════════════════════════
# Test targets
# ═══════════════════════════════════════════════════════════════════════════

test:
	docker run --rm --gpus all \
		-e NVIDIA_VISIBLE_DEVICES=0 \
		$(REGISTRY)/$(IMAGE_NAME):v$(VERSION) \
		python3 -c "from vllm.gonka_poc import PoCManagerV1; print('PoC V1 OK')"

# ═══════════════════════════════════════════════════════════════════════════
# Utility targets
# ═══════════════════════════════════════════════════════════════════════════

# Show image sizes
sizes:
	@echo "=== Image Sizes ==="
	@docker images $(REGISTRY)/$(IMAGE_NAME) --format "table {{.Repository}}:{{.Tag}}\t{{.Size}}"

# Clean build cache
clean:
	docker builder prune -f

# Shell into container
shell:
	docker run --rm -it --gpus all \
		$(REGISTRY)/$(IMAGE_NAME):v$(VERSION) \
		/bin/bash

# ═══════════════════════════════════════════════════════════════════════════
# Info
# ═══════════════════════════════════════════════════════════════════════════

info:
	@echo "Registry: $(REGISTRY)"
	@echo "Image:    $(IMAGE_NAME)"
	@echo "Version:  $(VERSION)"
	@echo "vLLM:     $(VLLM_VERSION)"
	@echo "CUDA:     $(CUDA_VERSION)"
	@echo ""
	@echo "Targets:"
	@echo "  make build  - Build image"
	@echo "  make push   - Push to registry"
	@echo "  make test   - Test image"
	@echo "  make shell  - Interactive shell"
	@echo "  make sizes  - Show image sizes"
