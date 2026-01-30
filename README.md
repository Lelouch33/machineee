# Machine PoC V1

PoC (Proof of Computation) module for vLLM V1 engine (vLLM 0.14.0+).

## Overview

This module provides PoC support for modern vLLM engines with V1 architecture and collective_rpc. It enables MLNode to integrate with vLLM for computation verification with improved performance.

## Key Features

- **V1 Engine Support**: Full compatibility with vLLM 0.14.0+ V1 architecture
- **EmbeddingInjectionHook**: Solves V1's input_ids requirement without breaking model forward pass
- **Multi-GPU Support**: TP (Tensor Parallel) and PP (Pipeline Parallel) synchronization
- **High Performance**: ~1291 nonces/min on B300 with seq_len=1024

## Requirements

- NVIDIA GPU:
  - Blackwell B300/B200 (SM 10.x) - use `install_b300.sh`
  - Hopper H100/H200 (SM 9.0) - use `install_universal.sh`
  - Ampere A100 (SM 8.0) - use `install_universal.sh`
- CUDA 13.0
- Python 3.12+
- vLLM 0.14.0+

## Installation

### Quick Install (Auto-detect GPU)

```bash
sudo bash install.sh
```

### Manual Install (B300 Blackwell)

```bash
sudo bash install_b300.sh
```

### Manual Install (H100/A100)

```bash
sudo bash install_universal.sh
```

## API Endpoints

Once installed, vLLM will expose PoC endpoints:

- `POST /api/v1/pow/init/generate` - Start continuous generation
- `POST /api/v1/pow/generate` - Generate artifacts for specific nonces
- `GET /api/v1/pow/status` - Get current status
- `POST /api/v1/pow/stop` - Stop generation

### Example: Start Generation

```bash
curl -X POST http://127.0.0.1:5000/api/v1/pow/init/generate \
  -H "Content-Type: application/json" \
  -d '{
    "block_hash": "ba5b00876d55f32856a5fbae8b4ebd79c5a90c763b23c15b7ec813cda54965ab",
    "block_height": 2401239,
    "public_key": "03b9be7b7bcc197eadcd6cd0ab62d129aa39b94e53ea72ecacc4d95ccec859435c",
    "node_id": 0,
    "node_count": 1,
    "batch_size": 32,
    "params": {
      "model": "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8",
      "seq_len": 1024,
      "k_dim": 12
    }
  }'
```

### Example: Generate Specific Nonces

```bash
curl -X POST http://127.0.0.1:5000/api/v1/pow/generate \
  -H "Content-Type: application/json" \
  -d '{
    "block_hash": "ba5b00876d55f32856a5fbae8b4ebd79c5a90c763b23c15b7ec813cda54965ab",
    "block_height": 2401239,
    "public_key": "03b9be7b7bcc197eadcd6cd0ab62d129aa39b94e53ea72ecacc4d95ccec859435c",
    "node_id": 0,
    "node_count": 1,
    "nonces": [0, 1, 2],
    "batch_size": 3,
    "wait": true,
    "params": {
      "model": "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8",
      "seq_len": 1024,
      "k_dim": 12
    }
  }'
```

## Module Structure

```
machineee/
├── install.sh              # Auto-detect GPU and install
├── install_b300.sh         # Blackwell B300/B200 install
├── install_universal.sh    # H100/H200/A100 install
├── README.md               # This file
├── gonka_poc/
│   ├── __init__.py         # Module exports
│   ├── config.py           # Configuration classes
│   ├── data.py             # Data structures and encoding
│   ├── gpu_random.py       # Deterministic GPU random generation
│   ├── layer_hooks.py      # Layer transformation hooks
│   ├── manager.py          # PoCManagerV1 for V1 engine
│   ├── poc_model_runner.py # Forward pass with EmbeddingInjectionHook
│   ├── routes.py           # FastAPI routes
│   └── validation.py       # Artifact validation
└── patches/
    └── runner_patch.md     # MLNode runner.py patch documentation
```

## V1 Engine Changes

### Key Differences from V0

1. **Input ID Requirement**: V1 requires `input_ids` as positional argument
2. **EmbeddingInjectionHook**: Intercepts `embed_tokens` to inject custom embeddings
3. **Enforce Eager Mode**: `--enforce-eager` flag required to avoid CUDA graph batch size limits
4. **Insecure Serialization**: Required for collective_rpc function passing

### MLNode runner.py Patch

The install scripts automatically patch `runner.py` with:

```python
# vllm_command additions
"--enforce-eager",

# Environment variables
env["VLLM_USE_V1"] = "1"
env["VLLM_USE_CUDA_GRAPHS"] = "0"
env["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
```

## Performance

| GPU | seq_len | batch_size | Rate (nonces/min) |
|-----|---------|------------|-------------------|
| B300 (2x) | 1024 | 32 | ~1291 |
| H100 (baseline V0) | 1024 | 32 | ~550 |

## Troubleshooting

### CUDA Graphs Error
```
AssertionError: Shape: 32768 out of considered ranges: [(1, 8192)]
```
**Solution**: Ensure `--enforce-eager` is in vllm_command

### Serialization Error
```
Object of type <class 'function'> is not serializable
```
**Solution**: Set `VLLM_ALLOW_INSECURE_SERIALIZATION=1`

### V1 Not Enabled
Check logs for:
```
Initializing a V1 LLM engine
```
If not present, verify `VLLM_USE_V1=1` in runner.py

## License

MIT
