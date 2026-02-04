# Docker Build для Gonka MLNode

Оптимизированные Docker образы для vLLM 0.14.0 + PoC V1 Engine.

## Сравнение размеров

| Образ | Оригинал | Оптимизированный | Экономия |
|-------|----------|------------------|----------|
| vLLM base | 39 GB | - | - |
| MLNode | 52 GB | ~12-15 GB | **~70%** |

## Почему меньше?

Оригинальный образ дублирует зависимости:
- PyTorch ставится дважды (~2GB x2)
- flash-attention ставится дважды (~2GB x2)
- scipy, fastapi, etc. дублируются

Наш подход:
- Ставим vLLM через `uv pip` (уже включает torch, scipy, etc.)
- Добавляем только уникальные зависимости MLNode (~100MB)
- Копируем только Python код packages (~10MB)

## Требования

- Docker с поддержкой NVIDIA GPU
- NVIDIA Driver 535+
- ~30GB места для сборки

## Сборка

```bash
cd docker

# Для H100/H200/A100
make build
```

## Структура

```
docker/
├── Dockerfile.universal  # H100/H200/A100 (TRITON_ATTN)
├── Makefile              # Build commands
└── README.md
```

## Запуск

```bash
# Запуск с GPU
docker run --gpus all -p 8080:8080 -p 5000:5000 \
    -v /root/.cache:/root/.cache \
    ghcr.io/lelouch33/gonka-mlnode:v0.14.0

# Интерактивный shell
make shell
```

## Переменные окружения

| Переменная | Значение | Описание |
|------------|----------|----------|
| VLLM_USE_V1 | 1 | V1 Engine |
| VLLM_ALLOW_INSECURE_SERIALIZATION | 1 | Для collective_rpc |
| HF_HOME | /root/.cache/huggingface | Кэш моделей |

## Push в registry

```bash
# Логин в GitHub Container Registry
echo $GITHUB_TOKEN | docker login ghcr.io -u USERNAME --password-stdin

# Push
make push
```
