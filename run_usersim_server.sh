#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# Launch a vLLM server hosting Qwen3.5-122B-A10B-FP8 as the tau2 user simulator.
#
# Deployed on its own GPUs / container / port so it runs alongside the Olmo-3
# agent vLLM without resource contention.
#
# Notes
# -----
# - --tensor-parallel-size 4 fits FP8 weights comfortably on 4×H100 80GB
#   (~30 GB/GPU for weights, ~40 GB/GPU free for KV cache after util=0.9).
# - --language-model-only skips the vision encoder (Qwen3.5 ships multimodal).
# - --default-chat-template-kwargs '{"enable_thinking": false}' disables the
#   default thinking mode server-side. The user simulator doesn't need
#   reasoning, and disabling it saves tokens + latency.
# - No reasoning parser (thinking is disabled).
# - --tool-call-parser qwen3_coder for the few tau2 domains where the user
#   simulator is also given tools (telecom, banking_knowledge).
# - --attention-backend FLASH_ATTN + VLLM_BATCH_INVARIANT=1 by default so the
#   user-sim is batch-invariant across runs, same as the Olmo container.
#   If Qwen MoE refuses to start with batch-invariance, pass --no-batch-invariant.
#
# Usage:
#   ./run_usersim_server.sh                            # defaults
#   ./run_usersim_server.sh --gpus 0,1,2,3 --port 8002 # custom GPUs / port
#   ./run_usersim_server.sh --no-wait                  # don't block on readiness
# ==============================================================================

GPUS="0,1,2,3"
PORT=8002
CONTAINER_NAME="ytahtah-vllm-usersim"
VLLM_IMAGE="vllm/vllm-openai:v0.19.0"
MODEL="Qwen/Qwen3.5-122B-A10B-FP8"
SERVE_NAME="Qwen/Qwen3.5-122B-A10B-FP8"
SHM_SIZE="16g"
TP_SIZE=""
# 65536 (64K) matches Olmo's max-model-len, so neither endpoint becomes the
# context bottleneck. Well within Qwen3.5's native 262144 (1/4 of trained max).
MAX_MODEL_LEN=65536
GPU_MEM_UTIL=0.9
# Batch invariance gives deterministic outputs across runs regardless of
# concurrent-request batch composition (needed when --max-concurrency > 1).
# Olmo already has it; without it on Qwen, user-sim outputs can flip on
# borderline tokens between runs even at temperature=0.
# Compatibility caveat: VLLM_BATCH_INVARIANT was validated mainly on dense
# models. Qwen3.5-122B-A10B is MoE; if startup fails, pass --no-batch-invariant.
BATCH_INVARIANT=true
# 30 min default — Qwen3.5-122B-A10B-FP8 weight load + CUDA-graph capture
# can easily take 15-25 min the first time, even after weights are cached.
MAX_WAIT=1800
NO_WAIT=false

HF_HOME="/mnt/nfs/ytahtah/hf_home"
VLLM_CACHE="/mnt/nfs/ytahtah/.cache/vllm_compile/usersim"
TMP_DIR="/mnt/nfs/ytahtah/tmp_compile/usersim"
TRITON_CACHE="/mnt/nfs/ytahtah/.triton_cache/usersim"

usage() {
    cat <<'USAGE'
Usage: ./run_usersim_server.sh [options]

Options:
  --model <hf_id>          HuggingFace model ID (default: Qwen/Qwen3.5-122B-A10B-FP8)
  --serve-name <name>      vLLM API name (default: same as --model)
  --gpus <ids>             Comma-separated GPU IDs (default: 0,1,2,3)
  --port <port>            Server port (default: 8002)
  --tp-size <N>            Tensor parallel size (default: number of GPUs)
  --max-model-len <N>      vLLM --max-model-len (default: 65536)
  --gpu-mem-util <F>       vLLM --gpu-memory-utilization (default: 0.9)
  --no-batch-invariant     Disable VLLM_BATCH_INVARIANT
                           (use this if Qwen MoE refuses to start with it on)
  --container-name <name>  Docker container name (default: ytahtah-vllm-usersim)
  --vllm-image <image>     vLLM Docker image (default: vllm/vllm-openai:v0.19.0)
  --shm-size <size>        Shared memory size (default: 16g)
  --vllm-cache-dir <dir>   Mounted as /root/.cache/vllm
                           (default: /mnt/nfs/ytahtah/.cache/vllm_compile/usersim)
  --tmp-dir <dir>          Mounted as /tmp
                           (default: /mnt/nfs/ytahtah/tmp_compile/usersim)
  --triton-cache-dir <dir> Mounted as /root/.triton
                           (default: /mnt/nfs/ytahtah/.triton_cache/usersim)
  --max-wait <seconds>     Max seconds to wait for readiness (default: 1800)
  --no-wait                Don't wait for the server to be ready
  -h, --help               Show this help
USAGE
    exit 0
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)             MODEL="$2"; shift 2 ;;
        --serve-name)        SERVE_NAME="$2"; shift 2 ;;
        --gpus)              GPUS="$2"; shift 2 ;;
        --port)              PORT="$2"; shift 2 ;;
        --tp-size)           TP_SIZE="$2"; shift 2 ;;
        --max-model-len)     MAX_MODEL_LEN="$2"; shift 2 ;;
        --gpu-mem-util)      GPU_MEM_UTIL="$2"; shift 2 ;;
        --no-batch-invariant) BATCH_INVARIANT=false; shift ;;
        --container-name)    CONTAINER_NAME="$2"; shift 2 ;;
        --vllm-image)        VLLM_IMAGE="$2"; shift 2 ;;
        --shm-size)          SHM_SIZE="$2"; shift 2 ;;
        --vllm-cache-dir)    VLLM_CACHE="$2"; shift 2 ;;
        --tmp-dir)           TMP_DIR="$2"; shift 2 ;;
        --triton-cache-dir)  TRITON_CACHE="$2"; shift 2 ;;
        --max-wait)          MAX_WAIT="$2"; shift 2 ;;
        --no-wait)           NO_WAIT=true; shift ;;
        -h|--help)           usage ;;
        *)                   echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

NUM_GPUS=$(echo "$GPUS" | tr ',' '\n' | wc -l)
TP_SIZE="${TP_SIZE:-$NUM_GPUS}"

echo "============================================================"
echo "vLLM User-Simulator Server (Qwen3.5)"
echo "============================================================"
echo "Model:            $MODEL"
echo "Serve name:       $SERVE_NAME"
echo "GPUs:             $GPUS ($NUM_GPUS GPUs, TP=$TP_SIZE)"
echo "Port:             $PORT"
echo "Container:        $CONTAINER_NAME"
echo "Image:            $VLLM_IMAGE"
echo "max_model_len:    $MAX_MODEL_LEN"
echo "gpu_mem_util:     $GPU_MEM_UTIL"
echo "Batch invariant:  $BATCH_INVARIANT"
echo "HF_HOME:          $HF_HOME"
echo "vLLM cache:       $VLLM_CACHE"
echo "tmp dir:          $TMP_DIR"
echo "Triton cache:     $TRITON_CACHE"
echo "============================================================"

mkdir -p "$VLLM_CACHE" "$TMP_DIR" "$TRITON_CACHE"

echo ""
echo "Stopping existing container '$CONTAINER_NAME' if any..."
docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

# Build the docker command as a string (required for the --gpus literal-quote
# trick: --gpus '"device=...,..."'). Same pattern as run_vllm_server.sh.
DOCKER_CMD="docker run -d"
DOCKER_CMD+=" --name ${CONTAINER_NAME}"
DOCKER_CMD+=" --gpus '\"device=${GPUS}\"'"
DOCKER_CMD+=" --shm-size ${SHM_SIZE}"
DOCKER_CMD+=" -p ${PORT}:8000"
DOCKER_CMD+=" -v ${HF_HOME}:/hf_home"
DOCKER_CMD+=" -v ${VLLM_CACHE}:/root/.cache/vllm"
DOCKER_CMD+=" -v ${TMP_DIR}:/tmp"
DOCKER_CMD+=" -v ${TRITON_CACHE}:/root/.triton"
DOCKER_CMD+=" -e HF_HOME=/hf_home"

if [[ -n "${HF_TOKEN:-}" ]]; then
    DOCKER_CMD+=" -e HF_TOKEN=${HF_TOKEN}"
fi

if [[ "$BATCH_INVARIANT" == "true" ]]; then
    DOCKER_CMD+=" -e VLLM_BATCH_INVARIANT=1"
fi

DOCKER_CMD+=" ${VLLM_IMAGE}"
DOCKER_CMD+=" --model ${MODEL}"
DOCKER_CMD+=" --served-model-name ${SERVE_NAME}"
DOCKER_CMD+=" --tensor-parallel-size ${TP_SIZE}"
DOCKER_CMD+=" --port 8000"
DOCKER_CMD+=" --trust-remote-code"
DOCKER_CMD+=" --language-model-only"
DOCKER_CMD+=" --max-model-len ${MAX_MODEL_LEN}"
DOCKER_CMD+=" --gpu-memory-utilization ${GPU_MEM_UTIL}"
# VLLM_BATCH_INVARIANT requires one of FLASH_ATTN, TRITON_ATTN, FLASH_ATTN_MLA,
# TRITON_MLA. Setting it unconditionally matches Olmo's container for parity.
DOCKER_CMD+=" --attention-backend FLASH_ATTN"
DOCKER_CMD+=" --default-chat-template-kwargs '{\"enable_thinking\": false}'"
# tau2's user_simulator passes tools to the LLM for domains that have
# user_tools (telecom and banking_knowledge currently). Without a tool-call
# parser, Qwen would emit tool calls as plain text, tau2 would see no
# structured tool_calls, and the user-tool flow would silently break.
# `qwen3_coder` is the parser the official Qwen3.5 vLLM recipe recommends.
DOCKER_CMD+=" --enable-auto-tool-choice --tool-call-parser qwen3_coder"

echo "Running: ${DOCKER_CMD}"
eval "${DOCKER_CMD}"

if [[ "$NO_WAIT" == "true" ]]; then
    echo ""
    echo "Container started. Skipping readiness check (--no-wait)."
    echo "Tail logs:  docker logs -f ${CONTAINER_NAME}"
    echo "Stop:       docker rm -f ${CONTAINER_NAME}"
    exit 0
fi

echo ""
echo "Waiting for server to be ready (downloading FP8 weights on first run can take a while)..."
WAITED=0
while ! curl -sf "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; do
    sleep 10
    WAITED=$((WAITED + 10))
    if [[ $WAITED -ge $MAX_WAIT ]]; then
        echo "ERROR: Server did not start within ${MAX_WAIT}s"
        echo "Last logs:"
        docker logs --tail 100 "$CONTAINER_NAME"
        exit 1
    fi
    echo "Waiting... (${WAITED}s)"
done

echo ""
echo "Server is ready on port ${PORT}."
echo "Test with:"
echo "  curl http://localhost:${PORT}/v1/models"
echo "  curl http://localhost:${PORT}/v1/chat/completions -H 'Content-Type: application/json' \\"
echo "    -d '{\"model\":\"${SERVE_NAME}\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":32}'"
echo "Tail logs:  docker logs -f ${CONTAINER_NAME}"
echo "Stop:       docker rm -f ${CONTAINER_NAME}"
