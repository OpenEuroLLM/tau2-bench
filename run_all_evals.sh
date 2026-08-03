#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# Run tau2-bench Olmo-3 evaluations sequentially across a list of checkpoints.
#
# For each checkpoint, deploys a vLLM server, runs all four tau2 domains with
# --num-trials trials, computes pass^k metrics, then tears down the server
# before moving to the next checkpoint.
#
# Output goes to:
#   data/simulations/olmo3_evals/eval_<name>/<domain>/results.json
#   data/simulations/olmo3_evals/eval_<name>/summary.txt
#
# Usage:
#   ./run_all_evals.sh                              # defaults: GPUs 4,5,6,7, 4 trials
#   ./run_all_evals.sh --gpus 0,1,2,3              # custom GPUs
#   ./run_all_evals.sh --trials 1                  # fast smoke run
#   ./run_all_evals.sh --checkpoints my_list.txt
#   ./run_all_evals.sh --domains retail,airline    # subset of domains
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RUN_EVAL="${SCRIPT_DIR}/run_eval.sh"

# Defaults
TRIALS=4
GPUS="4,5,6,7"
PORT=8000
SERVE_NAME="allenai/Olmo-3-7B-Instruct-SFT"
DOMAINS="retail,airline,telecom,banking_knowledge"
USER_LLM="hosted_vllm/Qwen/Qwen3.5-122B-A10B-FP8"
USER_SIM_HOST="localhost"
USER_SIM_PORT=8002
CHECKPOINTS_FILE="${SCRIPT_DIR}/checkpoints.txt"

usage() {
    cat <<'USAGE'
Usage: ./run_all_evals.sh [options]

Options:
  --trials N              Trials per task (default: 4)
  --gpus 0,1,2,3          GPU IDs (default: 4,5,6,7)
  --port 8000             vLLM server port (default: 8000)
  --serve-name NAME       vLLM API name (default: allenai/Olmo-3-7B-Instruct-SFT)
  --domains LIST          Comma-separated tau2 domains
                          (default: retail,airline,telecom,banking_knowledge)
  --user-llm MODEL        User simulator LLM
                          (default: hosted_vllm/Qwen/Qwen3.5-122B-A10B-FP8)
  --user-sim-host HOST    User-sim vLLM host (default: localhost)
  --user-sim-port PORT    User-sim vLLM port (default: 8002)
  --checkpoints FILE      Checkpoints list (default: ./checkpoints.txt)
  -h, --help              Show this help
USAGE
    exit 0
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --trials)       TRIALS="$2"; shift 2 ;;
        --gpus)         GPUS="$2"; shift 2 ;;
        --port)         PORT="$2"; shift 2 ;;
        --serve-name)   SERVE_NAME="$2"; shift 2 ;;
        --domains)      DOMAINS="$2"; shift 2 ;;
        --user-llm)         USER_LLM="$2"; shift 2 ;;
        --user-sim-host)    USER_SIM_HOST="$2"; shift 2 ;;
        --user-sim-port)    USER_SIM_PORT="$2"; shift 2 ;;
        --checkpoints)      CHECKPOINTS_FILE="$2"; shift 2 ;;
        -h|--help)      usage ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

[[ -f "$CHECKPOINTS_FILE" ]] || { echo "Checkpoints file not found: $CHECKPOINTS_FILE" >&2; exit 1; }
[[ -x "$RUN_EVAL" ]]         || { echo "run_eval.sh not found or not executable: $RUN_EVAL" >&2; exit 1; }

# Parse checkpoints (skip comment/blank lines)
mapfile -t CHECKPOINTS < <(grep -vE '^[[:space:]]*(#|$)' "$CHECKPOINTS_FILE")
N=${#CHECKPOINTS[@]}
(( N > 0 )) || { echo "No checkpoints in $CHECKPOINTS_FILE" >&2; exit 1; }

echo "════════════════════════════════════════════════════════════"
echo "Sequential tau2-bench eval across ${N} checkpoints"
echo "  GPUs:    ${GPUS}"
echo "  Port:    ${PORT}"
echo "  Trials:  ${TRIALS}"
echo "  Domains: ${DOMAINS}"
echo "  UserLLM: ${USER_LLM}"
echo "  UserSim: http://${USER_SIM_HOST}:${USER_SIM_PORT}/v1"
echo "════════════════════════════════════════════════════════════"

CURRENT=0
OK_NAMES=()
FAIL_NAMES=()

for entry in "${CHECKPOINTS[@]}"; do
    CURRENT=$((CURRENT + 1))
    IFS='|' read -r name model_path <<< "$entry"
    name="${name// /}"
    model_path="${model_path// /}"

    OUTPUT_DIR="olmo3_evals/eval_${name}"

    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "[${CURRENT}/${N}] ${name}"
    echo "  Model:  ${model_path}"
    echo "  Output: data/simulations/${OUTPUT_DIR}"
    echo "════════════════════════════════════════════════════════════"

    ARGS=(
        --model "${model_path}"
        --output-dir "${OUTPUT_DIR}"
        --num-trials "${TRIALS}"
        --gpus "${GPUS}"
        --port "${PORT}"
        --domains "${DOMAINS}"
        --user-llm "${USER_LLM}"
        --user-sim-host "${USER_SIM_HOST}"
        --user-sim-port "${USER_SIM_PORT}"
    )

    # Local checkpoint -> needs --serve-name; HF id -> default to model
    if [[ "${model_path}" == /* ]]; then
        ARGS+=(--serve-name "${SERVE_NAME}")
    fi

    if "$RUN_EVAL" "${ARGS[@]}"; then
        OK_NAMES+=("${name}")
        echo "[${CURRENT}/${N}] ✓ ${name}"
    else
        FAIL_NAMES+=("${name}")
        echo "[${CURRENT}/${N}] ✗ ${name}"
        # Ensure container is gone before continuing
        docker rm -f ytahtah-vllm-tau2 >/dev/null 2>&1 || true
    fi
done

echo ""
echo "════════════════════════════════════════════════════════════"
echo "All ${N} checkpoints processed."
echo "  ok:     ${#OK_NAMES[@]}  -> ${OK_NAMES[*]:-<none>}"
echo "  failed: ${#FAIL_NAMES[@]}  -> ${FAIL_NAMES[*]:-<none>}"
echo "════════════════════════════════════════════════════════════"
