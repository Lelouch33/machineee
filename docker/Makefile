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

.PHONY: build-universal build-b300 push clean

# Universal build (H100/H200/A100 with TRITON_ATTN)
build-universal:
	docker build \
		-f Dockerfile.universal \
		--build-arg CUDA_VERSION=$(CUDA_VERSION) \
		--build-arg VLLM_VERSION=$(VLLM_VERSION) \
		-t $(REGISTRY)/$(IMAGE_NAME):v$(VERSION)-universal \
		-t $(REGISTRY)/$(IMAGE_NAME):latest \
		..

# B300 Blackwell build (with FLASHINFER)
build-b300:
	docker build \
		-f Dockerfile.b300 \
		--build-arg CUDA_VERSION=$(CUDA_VERSION) \
		--build-arg VLLM_VERSION=$(VLLM_VERSION) \
		-t $(REGISTRY)/$(IMAGE_NAME):v$(VERSION)-b300 \
		..

# ═══════════════════════════════════════════════════════════════════════════
# Push targets
# ═══════════════════════════════════════════════════════════════════════════

push-universal:
	docker push $(REGISTRY)/$(IMAGE_NAME):v$(VERSION)-universal
	docker push $(REGISTRY)/$(IMAGE_NAME):latest

push-b300:
	docker push $(REGISTRY)/$(IMAGE_NAME):v$(VERSION)-b300

push: push-universal push-b300

# ═══════════════════════════════════════════════════════════════════════════
# Test targets
# ═══════════════════════════════════════════════════════════════════════════

test-universal:
	docker run --rm --gpus all \
		-e NVIDIA_VISIBLE_DEVICES=0 \
		$(REGISTRY)/$(IMAGE_NAME):v$(VERSION)-universal \
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
shell-universal:
	docker run --rm -it --gpus all \
		$(REGISTRY)/$(IMAGE_NAME):v$(VERSION)-universal \
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
	@echo "  make build-universal  - Build for H100/A100"
	@echo "  make build-b300       - Build for B300 Blackwell"
	@echo "  make push             - Push all images"
	@echo "  make test-universal   - Test universal image"
	@echo "  make sizes            - Show image sizes"
