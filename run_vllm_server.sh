#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# Launch a vLLM server in Docker for tau2-bench Olmo-3 evaluations.
#
# vLLM v0.19.0 has a built-in `olmo3` tool-call parser (Olmo3PythonicToolParser)
# that parses the model's <function_calls>...</function_calls> output into the
# standard OpenAI tool_calls schema. Combined with --enable-auto-tool-choice,
# tau2-bench can talk to this server via LiteLLM's hosted_vllm/ provider with
# no code changes.
#
# Parser limitation worth knowing about
# -------------------------------------
# When the model output contains a <function_calls>…</function_calls> block,
# vLLM's olmo3 parser keeps ONLY the inner block: any plain text before or
# after the tags is discarded and `content` in the returned ChatCompletion
# is set to None. e.g. an output like:
#     "Sure, let me check.<function_calls>foo()</function_calls> Anything else?"
# becomes tool_calls=[foo()], content=None — the wrapping prose is dropped.
# OLMo-3's training data emits tool calls on their own (no mixed prose), so
# this is unlikely in practice, but it does mean we can't preserve commentary
# alongside a tool call. If the model never emits a <function_calls> block,
# the full text passes through as `content` unchanged.
#
# Usage:
#   # HuggingFace model
#   ./run_vllm_server.sh --model allenai/Olmo-3-7B-Instruct-SFT
#
#   # Local checkpoint
#   ./run_vllm_server.sh --model /mnt/nfs/ytahtah/bfcl/phase2-a1-fc-sft \
#                        --serve-name allenai/Olmo-3-7B-Instruct-SFT
#
#   # Custom GPUs and port
#   ./run_vllm_server.sh --model allenai/Olmo-3-7B-Instruct-SFT \
#                        --gpus 0,1 --port 8010 --tp-size 2
# ==============================================================================

GPUS="4,5,6,7"
PORT=8000
CONTAINER_NAME="ytahtah-vllm-tau2"
VLLM_IMAGE="vllm/vllm-openai:v0.19.0"
SERVE_NAME=""
MODEL=""
BATCH_INVARIANT=true
SHM_SIZE="16g"
TP_SIZE=""
MAX_WAIT=600
TOOL_PARSER="olmo3"

HF_HOME="/mnt/nfs/ytahtah/hf_home"
VLLM_CACHE="/mnt/nfs/ytahtah/.cache/vllm_compile"
TMP_DIR="/mnt/nfs/ytahtah/tmp_compile"
TRITON_CACHE="/mnt/nfs/ytahtah/.triton_cache"

usage() {
    cat <<'USAGE'
Usage: ./run_vllm_server.sh --model <model> [options]

Required:
  --model <path|hf_id>       HuggingFace model ID or local checkpoint path

Options:
  --serve-name <name>        Model name for vLLM API (required for local checkpoints,
                             defaults to --model for HF models)
  --gpus <ids>               Comma-separated GPU IDs (default: 4,5,6,7)
  --port <port>              Server port (default: 8000)
  --tp-size <N>              Tensor parallel size (default: number of GPUs)
  --container-name <name>    Docker container name (default: ytahtah-vllm-tau2)
  --vllm-image <image>       vLLM Docker image (default: vllm/vllm-openai:v0.19.0)
  --shm-size <size>          Shared memory size (default: 16g)
  --tool-parser <name>       Tool-call parser passed to vLLM
                             (default: olmo3 — set "" to disable tool-call parsing)
  --no-batch-invariant       Disable VLLM_BATCH_INVARIANT
  --max-wait <seconds>       Max seconds to wait for server readiness (default: 600)
  --no-wait                  Don't wait for the server to be ready
  --vllm-cache-dir <dir>     Host dir mounted as /root/.cache/vllm
                             (default: /mnt/nfs/ytahtah/.cache/vllm_compile)
  --tmp-dir <dir>            Host dir mounted as /tmp
                             (default: /mnt/nfs/ytahtah/tmp_compile)
  --triton-cache-dir <dir>   Host dir mounted as /root/.triton
                             (default: /mnt/nfs/ytahtah/.triton_cache)
  -h, --help                 Show this help
USAGE
    exit 0
}

NO_WAIT=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)              MODEL="$2"; shift 2 ;;
        --serve-name)         SERVE_NAME="$2"; shift 2 ;;
        --gpus)               GPUS="$2"; shift 2 ;;
        --port)               PORT="$2"; shift 2 ;;
        --tp-size)            TP_SIZE="$2"; shift 2 ;;
        --container-name)     CONTAINER_NAME="$2"; shift 2 ;;
        --vllm-image)         VLLM_IMAGE="$2"; shift 2 ;;
        --shm-size)           SHM_SIZE="$2"; shift 2 ;;
        --tool-parser)        TOOL_PARSER="$2"; shift 2 ;;
        --no-batch-invariant) BATCH_INVARIANT=false; shift ;;
        --max-wait)           MAX_WAIT="$2"; shift 2 ;;
        --no-wait)            NO_WAIT=true; shift ;;
        --vllm-cache-dir)     VLLM_CACHE="$2"; shift 2 ;;
        --tmp-dir)            TMP_DIR="$2"; shift 2 ;;
        --triton-cache-dir)   TRITON_CACHE="$2"; shift 2 ;;
        -h|--help)            usage ;;
        *)                    echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "$MODEL" ]]; then
    echo "Error: --model is required" >&2
    usage
fi

NUM_GPUS=$(echo "$GPUS" | tr ',' '\n' | wc -l)
TP_SIZE="${TP_SIZE:-$NUM_GPUS}"

IS_LOCAL=false
if [[ -d "$MODEL" ]]; then
    IS_LOCAL=true
    MODEL="$(cd "$MODEL" && pwd)"
    if [[ -z "$SERVE_NAME" ]]; then
        echo "Error: --serve-name is required for local checkpoints" >&2
        exit 1
    fi
fi

SERVE_NAME="${SERVE_NAME:-$MODEL}"

echo "============================================================"
echo "vLLM Server (tau2-bench / Olmo-3)"
echo "============================================================"
echo "Model:            $MODEL"
echo "Serve name:       $SERVE_NAME"
echo "Local checkpoint: $IS_LOCAL"
echo "GPUs:             $GPUS ($NUM_GPUS GPUs, TP=$TP_SIZE)"
echo "Port:             $PORT"
echo "Tool parser:      ${TOOL_PARSER:-<none>}"
echo "Batch invariant:  $BATCH_INVARIANT"
echo "Container:        $CONTAINER_NAME"
echo "Image:            $VLLM_IMAGE"
echo "============================================================"

echo ""
echo "Stopping existing container '$CONTAINER_NAME' if any..."
docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

# Build docker command as a string. We use a string (not array) because
# --gpus requires literal inner quotes that bash arrays cannot represent:
#   --gpus '"device=4,5,6,7"'
DOCKER_CMD="docker run -d"
DOCKER_CMD+=" --name ${CONTAINER_NAME}"
DOCKER_CMD+=" --gpus '\"device=${GPUS}\"'"
DOCKER_CMD+=" --shm-size ${SHM_SIZE}"
DOCKER_CMD+=" -p ${PORT}:${PORT}"
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

if [[ "$IS_LOCAL" == "true" ]]; then
    DOCKER_CMD+=" -v ${MODEL}:/model"
fi

DOCKER_CMD+=" ${VLLM_IMAGE}"

if [[ "$IS_LOCAL" == "true" ]]; then
    DOCKER_CMD+=" --model /model --served-model-name ${SERVE_NAME}"
else
    DOCKER_CMD+=" --model ${MODEL} --served-model-name ${SERVE_NAME}"
fi

DOCKER_CMD+=" --tensor-parallel-size ${TP_SIZE}"
DOCKER_CMD+=" --port ${PORT}"
DOCKER_CMD+=" --attention-backend FLASH_ATTN"
DOCKER_CMD+=" --trust-remote-code"

# Enable native function-calling via the OpenAI-compatible Chat Completions API.
# vLLM's built-in olmo3 parser extracts <function_calls>func(arg=val)</function_calls>
# from the model output into structured tool_calls.
if [[ -n "$TOOL_PARSER" ]]; then
    DOCKER_CMD+=" --enable-auto-tool-choice --tool-call-parser ${TOOL_PARSER}"
fi

echo "Running: ${DOCKER_CMD}"
eval "${DOCKER_CMD}"

if [[ "$NO_WAIT" == "true" ]]; then
    echo ""
    echo "Container started. Skipping readiness check (--no-wait)."
    echo "Stop: docker rm -f ${CONTAINER_NAME}"
    exit 0
fi

echo ""
echo "Waiting for server to be ready..."
WAITED=0
while ! curl -sf "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; do
    sleep 5
    WAITED=$((WAITED + 5))
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
echo "Stop: docker rm -f ${CONTAINER_NAME}"
