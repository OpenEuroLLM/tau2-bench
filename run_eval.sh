#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# tau2-bench Olmo-3 Evaluation Runner
#
# Deploys a vLLM server (or reuses one) and runs `tau2 run` across all four
# tau2-bench domains (retail, airline, telecom, banking_knowledge) with N
# trials per task. Saves results to data/simulations/<output_dir>/<domain>/.
# At the end, aggregates pass^k metrics and writes a summary.txt.
#
# Tool-call parser note (--tool-parser olmo3, vLLM v0.19.0+)
# ----------------------------------------------------------
# When the model output contains a <function_calls>…</function_calls> block,
# vLLM's olmo3 parser keeps only the inner block: any plain text before or
# after the tags is discarded and the response `content` is set to None.
# OLMo-3's training data does not interleave prose with tool calls, so this
# is unlikely in practice, but it does mean mixed-content assistant messages
# are not preserved. If the model emits no <function_calls> block, the full
# text passes through as `content` unchanged.
#
# Usage:
#   # HuggingFace model
#   ./run_eval.sh --model allenai/Olmo-3-7B-Instruct-SFT --num-trials 4
#
#   # Local checkpoint
#   ./run_eval.sh --model /mnt/nfs/ytahtah/bfcl/phase2-a1-fc-sft \
#                 --serve-name allenai/Olmo-3-7B-Instruct-SFT \
#                 --output-dir eval_phase2-a1-fc-sft \
#                 --num-trials 4 --gpus 4,5,6,7
#
#   # Just specific domains
#   ./run_eval.sh --model allenai/Olmo-3-7B-Instruct-SFT \
#                 --domains retail,airline --num-trials 4
# ==============================================================================

# ── Defaults ──────────────────────────────────────────────────────────────────
NUM_TRIALS=4
DOMAINS="retail,airline,telecom,banking_knowledge"
GPUS="4,5,6,7"
PORT=8000
CONTAINER_NAME="ytahtah-vllm-tau2"
VLLM_IMAGE="vllm/vllm-openai:v0.19.0"
# User simulator is a separate local vLLM (Qwen3.5-122B-A10B-FP8) served by
# run_usersim_server.sh on GPUs 0-3 / port 8002. Both --user-llm-args and
# --agent-llm-args carry a per-call `api_base` so the agent (Olmo) and user
# (Qwen) talk to different local endpoints — no OpenAI dependency.
USER_LLM="hosted_vllm/Qwen/Qwen3.5-122B-A10B-FP8"
USER_SIM_HOST="localhost"
USER_SIM_PORT=8002
# USER_LLM_ARGS / AGENT_LLM_ARGS are built dynamically below from USER_SIM_*
# and PORT so the api_base always matches. Set to non-empty here to let
# `--user-llm-args` / `--agent-llm-args` override the auto-build entirely.
USER_LLM_ARGS=""
AGENT_LLM_ARGS=""
RETRIEVAL_CONFIG="bm25"
MAX_STEPS=200
# Concurrency bound by Olmo TP=4 throughput now, not OpenAI TPM. Bumped from
# 8 → 16 to try to extract more out of vLLM's continuous batching on a
# saturated GPU. Olmo's KV cache had headroom (~3.5% used at 8 concurrent),
# so 16 fits comfortably. Worst case it's neutral; best case ~30-50% speedup.
MAX_CONCURRENCY=16
# Task-level retries on top of LiteLLM's internal retries. Both endpoints are
# local now, so most retries will be unnecessary — kept generous as safety net.
MAX_RETRIES=10
RETRY_DELAY=10
SEED=300
OUTPUT_DIR=""
SERVE_NAME=""
MODEL=""
SKIP_SERVER=false
BATCH_INVARIANT=true
SHM_SIZE="16g"
TP_SIZE=""
TOOL_PARSER="olmo3"

# Paths
BFCL_DIR="$(cd "$(dirname "$0")" && pwd)"
HF_HOME="/mnt/nfs/ytahtah/hf_home"
VLLM_CACHE="/mnt/nfs/ytahtah/.cache/vllm_compile"
TMP_DIR="/mnt/nfs/ytahtah/tmp_compile"
TRITON_CACHE="/mnt/nfs/ytahtah/.triton_cache"

usage() {
    cat <<'USAGE'
Usage: ./run_eval.sh --model <model> [options]

Required:
  --model <path|hf_id>       HuggingFace model ID or local checkpoint path

Options:
  --serve-name <name>        Model name for vLLM API (required for local checkpoints,
                             defaults to --model value for HF models). This name is
                             what tau2 will use as the agent-llm: hosted_vllm/<name>
  --output-dir <dir>         Output directory under data/simulations/
                             (default: olmo3_evals/eval_<safe_serve_name>_<ts>)
  --domains <list>           Comma-separated tau2 domains
                             (default: retail,airline,telecom,banking_knowledge)
  --num-trials <N>           Trials per task (default: 4)
  --user-llm <model>         User-simulator model in LiteLLM format
                             (default: hosted_vllm/Qwen/Qwen3.5-122B-A10B-FP8)
  --user-sim-host <host>     Host where the user-sim vLLM listens (default: localhost)
  --user-sim-port <port>     Port where the user-sim vLLM listens (default: 8002)
  --user-llm-args <json>     JSON dict for user-llm args. If unset, built from
                             user-sim host/port as:
                             '{"temperature":0.0,"num_retries":10,"api_base":"http://<host>:<port>/v1"}'
                             If set, it overrides the auto-build entirely —
                             must include `api_base` to reach the user-sim vLLM.
  --agent-llm-args <json>    JSON dict for agent-llm args. If unset, built as:
                             '{"temperature":0.0,"num_retries":10,"api_base":"http://localhost:<--port>/v1"}'
  --retrieval-config <name>  Knowledge-domain retrieval config (default: bm25)
  --max-steps <N>            tau2 --max-steps (default: 200)
  --max-concurrency <N>      tau2 --max-concurrency (default: 16)
  --max-retries <N>          tau2 --max-retries (default: 10)
  --retry-delay <SEC>        tau2 --retry-delay seconds (default: 10)
  --seed <N>                 tau2 --seed (default: 300)
  --gpus <ids>               Comma-separated GPU IDs (default: 4,5,6,7)
  --port <port>              vLLM server port (default: 8000)
  --skip-server              Don't deploy vLLM (assume already running at --port)
  --tool-parser <name>       vLLM --tool-call-parser (default: olmo3)
  --no-batch-invariant       Disable VLLM_BATCH_INVARIANT
  --container-name <name>    Docker container name (default: ytahtah-vllm-tau2)
  --vllm-image <image>       vLLM Docker image (default: vllm/vllm-openai:v0.19.0)
  --tp-size <N>              Tensor parallel size (default: number of GPUs)
  --shm-size <size>          Shared memory size (default: 16g)
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

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)             MODEL="$2"; shift 2 ;;
        --serve-name)        SERVE_NAME="$2"; shift 2 ;;
        --output-dir)        OUTPUT_DIR="$2"; shift 2 ;;
        --domains)           DOMAINS="$2"; shift 2 ;;
        --num-trials)        NUM_TRIALS="$2"; shift 2 ;;
        --user-llm)          USER_LLM="$2"; shift 2 ;;
        --user-sim-host)     USER_SIM_HOST="$2"; shift 2 ;;
        --user-sim-port)     USER_SIM_PORT="$2"; shift 2 ;;
        --user-llm-args)     USER_LLM_ARGS="$2"; shift 2 ;;
        --agent-llm-args)    AGENT_LLM_ARGS="$2"; shift 2 ;;
        --retrieval-config)  RETRIEVAL_CONFIG="$2"; shift 2 ;;
        --max-steps)         MAX_STEPS="$2"; shift 2 ;;
        --max-concurrency)   MAX_CONCURRENCY="$2"; shift 2 ;;
        --max-retries)       MAX_RETRIES="$2"; shift 2 ;;
        --retry-delay)       RETRY_DELAY="$2"; shift 2 ;;
        --seed)              SEED="$2"; shift 2 ;;
        --gpus)              GPUS="$2"; shift 2 ;;
        --port)              PORT="$2"; shift 2 ;;
        --skip-server)       SKIP_SERVER=true; shift ;;
        --tool-parser)       TOOL_PARSER="$2"; shift 2 ;;
        --no-batch-invariant) BATCH_INVARIANT=false; shift ;;
        --container-name)    CONTAINER_NAME="$2"; shift 2 ;;
        --vllm-image)        VLLM_IMAGE="$2"; shift 2 ;;
        --tp-size)           TP_SIZE="$2"; shift 2 ;;
        --shm-size)          SHM_SIZE="$2"; shift 2 ;;
        --vllm-cache-dir)    VLLM_CACHE="$2"; shift 2 ;;
        --tmp-dir)           TMP_DIR="$2"; shift 2 ;;
        --triton-cache-dir)  TRITON_CACHE="$2"; shift 2 ;;
        -h|--help)           usage ;;
        *)                   echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "$MODEL" ]]; then
    echo "Error: --model is required" >&2
    usage
fi

# ── Derived values ────────────────────────────────────────────────────────────
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
AGENT_LLM="hosted_vllm/${SERVE_NAME}"

# Build api_base URLs from --port (Olmo) and --user-sim-{host,port} (Qwen).
OLMO_API_BASE="http://localhost:${PORT}/v1"
USER_SIM_API_BASE="http://${USER_SIM_HOST}:${USER_SIM_PORT}/v1"

# Build USER_LLM_ARGS / AGENT_LLM_ARGS unless the caller already supplied them.
# (If they did, we trust them — but they must include api_base themselves.)
#
# max_tokens caps mirror BFCL's base_oss_handler (cap output at 4096 per call).
# Without this, Olmo can ramble for hours within a single agent step, holding
# up the whole sim (we observed a 2+ hour hang). Same cap for both sides —
# Qwen's user-sim utterances are typically short, so 4096 is loose headroom.
if [[ -z "$USER_LLM_ARGS" ]]; then
    USER_LLM_ARGS=$(printf '{"temperature": 0.0, "num_retries": 10, "max_tokens": 4096, "api_base": "%s"}' "$USER_SIM_API_BASE")
fi
if [[ -z "$AGENT_LLM_ARGS" ]]; then
    AGENT_LLM_ARGS=$(printf '{"temperature": 0.0, "num_retries": 10, "max_tokens": 4096, "api_base": "%s"}' "$OLMO_API_BASE")
fi

# Output dir under data/simulations/ for tau2's --save-to.
# tau2 saves to: DATA_DIR/simulations/<save-to>/results.json
if [[ -z "$OUTPUT_DIR" ]]; then
    SAFE_NAME=$(echo "$SERVE_NAME" | tr '/' '_')
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    OUTPUT_DIR="olmo3_evals/eval_${SAFE_NAME}_${TIMESTAMP}"
fi

# ── Preflight: verify tau2 env + required extras, and resolve DATA_DIR ────────
# `banking_knowledge` needs the `knowledge` extra (rank_bm25). Bail early with
# a clear remediation if it's missing, since failure deep in tau2 is opaque.
PREFLIGHT_OUTPUT=$(cd "$BFCL_DIR" && uv run python -c "
import sys
try:
    from tau2.utils.utils import DATA_DIR
except Exception as e:
    print('TAU2_IMPORT_ERROR', e); sys.exit(2)
print(f'DATA_DIR={DATA_DIR}')
needs_knowledge = '$DOMAINS'.find('banking_knowledge') >= 0
if needs_knowledge:
    try:
        import rank_bm25  # noqa: F401
    except ImportError:
        print('MISSING_KNOWLEDGE_EXTRA'); sys.exit(3)
" 2>&1)
PREFLIGHT_STATUS=$?
if [[ $PREFLIGHT_STATUS -ne 0 ]]; then
    echo "ERROR: preflight failed" >&2
    echo "$PREFLIGHT_OUTPUT" >&2
    if echo "$PREFLIGHT_OUTPUT" | grep -q MISSING_KNOWLEDGE_EXTRA; then
        echo "" >&2
        echo "  Fix: cd $BFCL_DIR && uv sync --extra knowledge" >&2
        echo "  (or drop banking_knowledge from --domains)" >&2
    fi
    exit 1
fi

TAU2_DATA_DIR_RESOLVED=$(echo "$PREFLIGHT_OUTPUT" | grep '^DATA_DIR=' | sed 's/^DATA_DIR=//')
if [[ -z "$TAU2_DATA_DIR_RESOLVED" ]]; then
    echo "WARNING: could not resolve tau2 DATA_DIR; falling back to ${BFCL_DIR}/data" >&2
    TAU2_DATA_DIR_RESOLVED="${BFCL_DIR}/data"
fi

# Verify the user-simulator vLLM is reachable. Don't try to start it here —
# it's a long-lived service launched separately by run_usersim_server.sh.
if ! curl -sf "${USER_SIM_API_BASE}/models" >/dev/null 2>&1; then
    echo "ERROR: user-simulator vLLM not reachable at ${USER_SIM_API_BASE}" >&2
    echo "  Start it with: ./run_usersim_server.sh" >&2
    echo "  Or skip the check with --user-sim-host/--user-sim-port if it's elsewhere." >&2
    exit 1
fi

# Absolute output path (where summary.txt lives, and where tau2 writes results)
ABS_OUTPUT="${TAU2_DATA_DIR_RESOLVED}/simulations/${OUTPUT_DIR}"

mkdir -p "$ABS_OUTPUT" "$VLLM_CACHE" "$TMP_DIR" "$TRITON_CACHE"

echo "============================================================"
echo "tau2-bench Olmo-3 Evaluation"
echo "============================================================"
echo "Model:            $MODEL"
echo "Serve name:       $SERVE_NAME"
echo "Local checkpoint: $IS_LOCAL"
echo "Agent LLM:        $AGENT_LLM"
echo "  api_base:       $OLMO_API_BASE"
echo "User LLM:         $USER_LLM"
echo "  api_base:       $USER_SIM_API_BASE"
echo "User LLM args:    $USER_LLM_ARGS"
echo "Agent LLM args:   $AGENT_LLM_ARGS"
echo "Domains:          $DOMAINS"
echo "Trials/task:      $NUM_TRIALS"
echo "Max steps:        $MAX_STEPS"
echo "Concurrency:      $MAX_CONCURRENCY"
echo "Max retries:      $MAX_RETRIES"
echo "Retry delay:      ${RETRY_DELAY}s"
echo "Seed:             $SEED"
echo "GPUs:             $GPUS ($NUM_GPUS GPUs, TP=$TP_SIZE)"
echo "Port:             $PORT"
echo "Tool parser:      ${TOOL_PARSER:-<none>}"
echo "Batch invariant:  $BATCH_INVARIANT"
echo "Output:           $ABS_OUTPUT"
echo "  (tau2 --save-to: $OUTPUT_DIR)"
echo "============================================================"

# ── Deploy vLLM server ────────────────────────────────────────────────────────
deploy_server() {
    echo ""
    echo "[server] Stopping existing container '$CONTAINER_NAME' if any..."
    docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

    echo "[server] Starting vLLM server..."

    local DOCKER_CMD="docker run -d"
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

    if [[ -n "$TOOL_PARSER" ]]; then
        DOCKER_CMD+=" --enable-auto-tool-choice --tool-call-parser ${TOOL_PARSER}"
    fi

    echo "[server] Running: ${DOCKER_CMD}"
    eval "${DOCKER_CMD}"

    echo "[server] Waiting for server to be ready..."
    local MAX_WAIT=600
    local WAITED=0
    while ! curl -sf "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; do
        sleep 5
        WAITED=$((WAITED + 5))
        if [[ $WAITED -ge $MAX_WAIT ]]; then
            echo "[server] ERROR: Server did not start within ${MAX_WAIT}s"
            echo "[server] Last logs:"
            docker logs --tail 100 "$CONTAINER_NAME"
            exit 1
        fi
        echo "[server] Waiting... (${WAITED}s)"
    done
    echo "[server] Server is ready."
}

if [[ "$SKIP_SERVER" == "false" ]]; then
    deploy_server
else
    echo "[server] Skipping server deployment (--skip-server)"
    # Quick sanity check that something IS at that port
    if ! curl -sf "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; then
        echo "[server] WARNING: nothing reachable at http://localhost:${PORT}/v1/models" >&2
    fi
fi

# ── Wire LiteLLM env to the two local vLLM endpoints ──────────────────────────
# Agent (Olmo) and user (Qwen) pass per-call api_base in their args, so neither
# strictly needs HOSTED_VLLM_API_BASE. We still set it to Olmo's URL as a
# defensive fallback for any hosted_vllm call without an explicit api_base.
export HOSTED_VLLM_API_BASE="$OLMO_API_BASE"
# LiteLLM requires SOME api_key value for hosted_vllm. Any non-empty string works.
export HOSTED_VLLM_API_KEY="${HOSTED_VLLM_API_KEY:-EMPTY}"
# config.py reads this at import time to route the NL_ASSERTIONS evaluator at
# the user-sim vLLM (so retail NL evals never call out to OpenAI).
export TAU2_USERSIM_API_BASE="$USER_SIM_API_BASE"

# ── Run tau2 across all requested domains ─────────────────────────────────────
IFS=',' read -r -a DOMAIN_LIST <<< "$DOMAINS"

# Build the entries that scripts/olmo3_summary.py expects:
#   retail=data/simulations/.../retail/results.json
SUMMARY_ENTRIES=()
RAN_OK=()
RAN_FAIL=()

for DOMAIN in "${DOMAIN_LIST[@]}"; do
    DOMAIN="${DOMAIN// /}"
    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "Domain: ${DOMAIN}"
    echo "════════════════════════════════════════════════════════════"

    SAVE_TO="${OUTPUT_DIR}/${DOMAIN}"
    RESULTS_PATH="${TAU2_DATA_DIR_RESOLVED}/simulations/${SAVE_TO}/results.json"

    # Build tau2 run command
    TAU2_ARGS=(
        run
        --domain "$DOMAIN"
        --agent llm_agent
        --agent-llm "$AGENT_LLM"
        --agent-llm-args "$AGENT_LLM_ARGS"
        --user user_simulator
        --user-llm "$USER_LLM"
        --user-llm-args "$USER_LLM_ARGS"
        --num-trials "$NUM_TRIALS"
        --max-steps "$MAX_STEPS"
        --max-concurrency "$MAX_CONCURRENCY"
        --max-retries "$MAX_RETRIES"
        --retry-delay "$RETRY_DELAY"
        --seed "$SEED"
        --save-to "$SAVE_TO"
        --auto-resume
        --log-level WARNING
    )

    if [[ "$DOMAIN" == "banking_knowledge" ]]; then
        TAU2_ARGS+=( --retrieval-config "$RETRIEVAL_CONFIG" )
    fi

    if (cd "$BFCL_DIR" && uv run tau2 "${TAU2_ARGS[@]}"); then
        RAN_OK+=("$DOMAIN")
    else
        echo "[${DOMAIN}] tau2 run failed (exit $?). Continuing with other domains."
        RAN_FAIL+=("$DOMAIN")
    fi

    SUMMARY_ENTRIES+=( "${DOMAIN}=${RESULTS_PATH}" )
done

# ── Aggregate metrics ─────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════"
echo "Aggregating metrics"
echo "════════════════════════════════════════════════════════════"

SUMMARY_PATH="${ABS_OUTPUT}/summary.txt"
(cd "$BFCL_DIR" && uv run python scripts/olmo3_summary.py \
    --output "$SUMMARY_PATH" "${SUMMARY_ENTRIES[@]}") || true

echo ""
echo "Domains OK:     ${RAN_OK[*]:-<none>}"
echo "Domains FAILED: ${RAN_FAIL[*]:-<none>}"
echo "Summary:        ${SUMMARY_PATH}"

# ── Cleanup ───────────────────────────────────────────────────────────────────
if [[ "$SKIP_SERVER" == "false" ]]; then
    echo ""
    echo "[server] Stopping container '$CONTAINER_NAME'..."
    docker stop "$CONTAINER_NAME" >/dev/null 2>&1 || true
    docker rm "$CONTAINER_NAME" >/dev/null 2>&1 || true
fi

echo ""
echo "============================================================"
echo "Done. Results under: ${ABS_OUTPUT}"
echo "============================================================"

# Exit non-zero only if every domain failed
if [[ ${#RAN_OK[@]} -eq 0 ]]; then
    exit 1
fi
