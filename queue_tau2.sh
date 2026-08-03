#!/usr/bin/env bash
set -uo pipefail

# ==============================================================================
# queue_tau2.sh  —  unified tau2-bench scheduler / queue
#
# One script that schedules the whole tau2-bench eval run:
#
#   1. Wait until ALL ${EXPECTED_GPUS} GPUs are free, continuously, for
#      ${STABLE_SECONDS}s. Detection is purely GPU-based (compute processes +
#      memory) — we DO NOT watch any container names. Whatever is holding the
#      GPUs (our BFCL, another user's training job, an idle-but-reserved vLLM
#      container from another team) is treated the same: we just wait for the
#      GPUs to free on their own. We never kill anything that isn't ours.
#   2. Launch the Qwen user-sim (run_usersim_server.sh) and BLOCK until its
#      /v1/models endpoint is ready. Abort the whole run if it never comes up.
#   3. Run the full eval sweep (run_all_evals.sh) across every checkpoint in
#      checkpoints.txt, all domains, with the user-sim serving the user turns.
#   4. Remove the user-sim container at the end of all the evals.
#
# This supersedes the old queue_tau2_after_bfcl.sh (which watched BFCL
# container names) and queue_tau2_after_gpus_free.sh — there is now a single
# canonical queue.
#
# Detection model (Phase 1)
#   Poll nvidia-smi every POLL_SECONDS. GPUs count as FREE only when:
#     (a) there are ZERO compute processes across all GPUs, AND
#     (b) every GPU's used memory is below MEM_FREE_THRESHOLD_MIB, AND
#     (c) nvidia-smi reports at least EXPECTED_GPUS GPUs.
#   "Busy if ANY signal says busy" is deliberately conservative — we never
#   launch on top of someone else's job because of a noisy reading. The free
#   streak must last STABLE_SECONDS continuously; ANY busy observation (or a
#   failed/ambiguous nvidia-smi query) resets the timer to zero.
#
#   The (a)+(b) pair guards a multi-user gotcha: on some nodes
#   `--query-compute-apps` lists only YOUR OWN processes, so a foreign job is
#   invisible there — but its memory still shows in --query-gpu, so (b) catches
#   it. And note an idle vLLM container still RESERVES its full
#   gpu_memory_utilization (~73 GB/GPU); it reads BUSY the whole time it is up,
#   so this waits for it to be REMOVED, not merely to go idle.
#
# Usage:
#   tmux new -s tau2_run
#   ./queue_tau2.sh
#   # detach with Ctrl-b d ; reattach with: tmux attach -t tau2_run
#
# Options:
#   --stable-seconds N   continuous free window required (default: 120)
#   --threshold-mib N    per-GPU "busy" memory threshold MiB (default: 2000)
#   --expected-gpus N    require this many GPUs visible & free (default: 8)
#   --poll-seconds N     nvidia-smi poll interval seconds (default: 15)
#   -h, --help           show this header
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

POLL_SECONDS=15
STABLE_SECONDS=120           # all GPUs free this long (continuous) => launch
MEM_FREE_THRESHOLD_MIB=2000  # a GPU using more than this is "busy". Idle H100
                             # sits near ~0; real jobs use many GB, so 2 GiB
                             # cleanly separates idle from active.
EXPECTED_GPUS=8              # tau2 needs all 8 (Qwen user-sim on 4 + Olmo on 4).
                             # Refuse to proceed unless nvidia-smi sees at least
                             # this many GPUs AND every one of them is free —
                             # guards against a partial query reading "free"
                             # when GPUs are merely missing from the listing.
USERSIM_NAME="ytahtah-vllm-usersim"

while [[ $# -gt 0 ]]; do
    case $1 in
        --stable-seconds)  STABLE_SECONDS="$2"; shift 2 ;;
        --threshold-mib)   MEM_FREE_THRESHOLD_MIB="$2"; shift 2 ;;
        --expected-gpus)   EXPECTED_GPUS="$2"; shift 2 ;;
        --poll-seconds)    POLL_SECONDS="$2"; shift 2 ;;
        -h|--help)         grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

mkdir -p "${SCRIPT_DIR}/logs"
LOG="${SCRIPT_DIR}/logs/queued_tau2_$(date +%Y%m%d_%H%M%S).log"

log() { printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "$LOG"; }

# Echo a single status token describing GPU occupancy:
#   "FREE maxmem=<n>MiB ngpu=<n>"  all visible GPUs idle, enough of them
#   "BUSY procs=<n>"               at least one compute process running
#   "BUSY maxmem=<n>MiB"           a GPU's memory exceeds the threshold
#   "BUSY ngpu=<n>/<exp>"          fewer GPUs visible than EXPECTED_GPUS
#   "UNKNOWN <reason>"             nvidia-smi failed — treated as busy (safe)
# Only "FREE ..." counts as free; everything else resets the free streak.
gpu_state() {
    local proc_out rc nproc mem_out maxmem=0 ngpu=0 m

    proc_out=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)
    rc=$?
    if (( rc != 0 )); then
        echo "UNKNOWN nvidia-smi(compute-apps)_rc=${rc}"
        return
    fi
    # Count lines containing a digit (a PID). Empty output => 0 processes.
    nproc=$(printf '%s\n' "$proc_out" | grep -c '[0-9]')
    if (( nproc > 0 )); then
        echo "BUSY procs=${nproc}"
        return
    fi

    mem_out=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)
    rc=$?
    if (( rc != 0 )); then
        echo "UNKNOWN nvidia-smi(memory)_rc=${rc}"
        return
    fi
    while read -r m; do
        m="${m//[[:space:]]/}"
        [[ -z "$m" ]] && continue
        # Guard against non-numeric lines (shouldn't happen, but be safe).
        [[ "$m" =~ ^[0-9]+$ ]] || { echo "UNKNOWN mem_parse='${m}'"; return; }
        ngpu=$(( ngpu + 1 ))
        (( m > maxmem )) && maxmem=$m
    done <<< "$mem_out"

    if (( ngpu < EXPECTED_GPUS )); then
        echo "BUSY ngpu=${ngpu}/${EXPECTED_GPUS}"
        return
    fi
    if (( maxmem > MEM_FREE_THRESHOLD_MIB )); then
        echo "BUSY maxmem=${maxmem}MiB"
        return
    fi
    echo "FREE maxmem=${maxmem}MiB ngpu=${ngpu}"
}

log "queue_tau2.sh starting"
log "Waiting for ALL ${EXPECTED_GPUS} GPUs free for ${STABLE_SECONDS}s continuous"
log "  (free = 0 compute procs AND every GPU < ${MEM_FREE_THRESHOLD_MIB}MiB used AND >= ${EXPECTED_GPUS} GPUs visible)"
log "Polling every ${POLL_SECONDS}s; combined log at ${LOG}"
log "NOTE: detection is GPU-only — no container names are watched."
log "NOTE: this script never kills GPU processes — it only waits for them to end."

# ── Phase 1: wait for all GPUs free for STABLE_SECONDS continuous ────────────
free_since=""    # epoch when the current free streak began; "" = not free now

while true; do
    now=$(date +%s)
    state=$(gpu_state)

    if [[ "$state" == FREE* ]]; then
        if [[ -z "$free_since" ]]; then
            free_since=$now
            log "GPUs FREE (${state}) — starting ${STABLE_SECONDS}s timer"
        fi
        free_for=$(( now - free_since ))
        log "GPUs free for ${free_for}s / ${STABLE_SECONDS}s (${state})"
        if (( free_for >= STABLE_SECONDS )); then
            log "GPUs free ${STABLE_SECONDS}s continuous — proceeding to launch"
            break
        fi
    else
        if [[ -n "$free_since" ]]; then
            log "GPUs busy again (${state}) — resetting free timer"
        else
            log "GPUs busy (${state})"
        fi
        free_since=""
    fi
    sleep "$POLL_SECONDS"
done

# ── Phase 2: launch the Qwen user-sim and wait for it to be ready ────────────
# run_usersim_server.sh runs `docker run -d` then BLOCKS until ytahtah-vllm-usersim's
# /v1/models endpoint responds, or exits non-zero on timeout. If it fails to
# come up, abort the whole run — we must not eval against a dead user-sim.
log "launching Qwen user-sim via ./run_usersim_server.sh (blocks until ready)"
"${SCRIPT_DIR}/run_usersim_server.sh" >> "$LOG" 2>&1
usersim_boot_status=$?
if (( usersim_boot_status != 0 )); then
    log "ERROR: run_usersim_server.sh exited with status ${usersim_boot_status}"
    log "queue_tau2.sh aborting — user-sim did not come up"
    exit 1
fi
log "Qwen user-sim ready"

# Brief pause so the Qwen warmup settles before tau2 starts hammering it
sleep 10

# ── Phase 3: run the full tau2 eval sweep ────────────────────────────────────
# run_all_evals.sh iterates every checkpoint in checkpoints.txt, all domains,
# with --auto-resume so an interrupted run can be re-launched and continue.
log "launching ./run_all_evals.sh (all checkpoints in checkpoints.txt)"
"${SCRIPT_DIR}/run_all_evals.sh" >> "$LOG" 2>&1
tau2_status=$?
log "run_all_evals.sh exited with status ${tau2_status}"

# Brief pause for any per-checkpoint Olmo container teardown to finish
sleep 15

# ── Phase 4: remove the user-sim at the end of all the evals ─────────────────
# Best effort — if it's already gone (e.g. crashed), nothing to do.
if docker ps --format '{{.Names}}' 2>/dev/null | grep -qx "$USERSIM_NAME"; then
    log "removing ${USERSIM_NAME}"
    if docker rm -f "$USERSIM_NAME" >/dev/null 2>&1; then
        log "${USERSIM_NAME} removed"
    else
        log "WARN: failed to remove ${USERSIM_NAME} — manual cleanup may be needed"
    fi
else
    log "${USERSIM_NAME} not present, nothing to remove"
fi

log "queue_tau2.sh complete (tau2=${tau2_status})"
