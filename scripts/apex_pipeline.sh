#!/usr/bin/env bash
#
# apex_pipeline.sh — Full APEX quantization pipeline for MoE models
#
# Orchestrates the complete flow: config → download → convert → quantize →
# imatrix → I-variants → eval → plots → publish
#
# Usage:
#   ./scripts/apex_pipeline.sh --config models/gemma4.yaml                    # full pipeline
#   ./scripts/apex_pipeline.sh --config models/gemma4.yaml --phase eval       # only eval phase
#   ./scripts/apex_pipeline.sh --config models/gemma4.yaml --from imatrix     # resume from imatrix
#   ./scripts/apex_pipeline.sh --config models/gemma4.yaml --skip publish     # skip publish
#   ./scripts/apex_pipeline.sh --config models/gemma4.yaml --only "quantize,eval"
#   ./scripts/apex_pipeline.sh --config models/gemma4.yaml --eval-suite native  # native only
#   ./scripts/apex_pipeline.sh --config models/gemma4.yaml --eval-suite all     # native + harness
#
# Phases (in order):
#   1. config         — Generate tensor-type configs for all profiles
#   2. download       — Download source model + baseline GGUFs for comparison
#   3. convert        — Convert safetensors to F16 GGUF (skip if source is already GGUF)
#   4. baseline       — Generate F16 reference logits, eval Q8_0 from baselines
#   5. quantize       — Create APEX Quality/Balanced/Compact/Mini quants
#   6. imatrix        — Generate importance matrix from calibration data
#   7. ivariants      — Create I-Quality/I-Balanced/I-Compact + I-Mini with imatrix
#   8. eval           — Run eval suite on all APEX quants
#   9. eval_baselines — Eval downloaded baseline GGUFs
#  10. plots          — Generate comparison plots
#  11. publish        — Upload GGUFs to HuggingFace
#  12. mlx             — Convert APEX profiles to MLX format (optional, requires mlx-lm)
#
# Eval suites (--eval-suite):
#   all     — PPL, KL, speed (native) + MMLU/ARC/TruthfulQA/HumanEval/MBPP/IFEval (harness) [DEFAULT]
#   native  — PPL, KL, HellaSwag, Winogrande, MMLU, ARC, TruthfulQA, speed (zero-shot, native only)
#   harness — HumanEval, MBPP, IFEval, MMLU, ARC, TruthfulQA (via lm-evaluation-harness only)
#
# Environment:
#   LLAMA_CPP_DIR     Path to llama.cpp build/bin (auto-detected)
#   WORK_DIR          Working directory for models (default: /workspace/data/apex)
#   NGL               GPU layers (default: 99)
#   PORT              Server port for harness evals (default: 8192)
#
set -euo pipefail

# Required for lm-eval-harness code execution (HumanEval, MBPP)
export HF_ALLOW_CODE_EVAL=1

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# ─── Defaults ──────────────────────────────────────────────────────────────────
CONFIG_FILE=""
PHASE=""
FROM_PHASE=""
SKIP_PHASES=""
ONLY_PHASES=""
EVAL_SUITE=""           # override from CLI; otherwise use YAML
DRY_RUN=false
NGL="${NGL:-99}"
WORK_DIR="${WORK_DIR:-/workspace/data/apex}"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-}"
PORT="${PORT:-8192}"

# ─── Parse args ────────────────────────────────────────────────────────────────
while [ $# -gt 0 ]; do
    case "$1" in
        --config|-c)      CONFIG_FILE="$2"; shift 2 ;;
        --phase|-p)       PHASE="$2"; shift 2 ;;
        --from)           FROM_PHASE="$2"; shift 2 ;;
        --skip)           SKIP_PHASES="$2"; shift 2 ;;
        --only)           ONLY_PHASES="$2"; shift 2 ;;
        --work-dir)       WORK_DIR="$2"; shift 2 ;;
        --eval-suite)     EVAL_SUITE="$2"; shift 2 ;;
        --dry-run)        DRY_RUN=true; shift ;;
        --help|-h)
            sed -n '3,42p' "$0"
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

[ -z "$CONFIG_FILE" ] && { echo "Error: --config required" >&2; exit 1; }
[ -f "$CONFIG_FILE" ] || { echo "Error: config not found: $CONFIG_FILE" >&2; exit 1; }

# ─── Parse YAML config ────────────────────────────────────────────────────────
parse_yaml() {
    local file="$1" key="$2"
    grep "^${key}:" "$file" 2>/dev/null | sed "s/^${key}:[[:space:]]*//" | sed 's/[[:space:]]*$//' | sed 's/^"//' | sed 's/"$//' || true
}

parse_yaml_list() {
    local file="$1" key="$2"
    awk "/^${key}:/{found=1; next} found && /^  - /{gsub(/^  - /,\"\"); print; next} found && /^[^ ]/{found=0}" "$file"
}

# ─── Load config ───────────────────────────────────────────────────────────────
MODEL_NAME=$(parse_yaml "$CONFIG_FILE" "name")
MODEL_ID=$(parse_yaml "$CONFIG_FILE" "model_id")
LAYERS=$(parse_yaml "$CONFIG_FILE" "layers")
HF_REPO=$(parse_yaml "$CONFIG_FILE" "hf_repo")
SOURCE_FORMAT=$(parse_yaml "$CONFIG_FILE" "source_format")
SOURCE_GGUF=$(parse_yaml "$CONFIG_FILE" "source_gguf")
CONFIG_PREFIX=$(parse_yaml "$CONFIG_FILE" "config_prefix")
TOKENIZER=$(parse_yaml "$CONFIG_FILE" "tokenizer")
SKIP_Q8_BASELINE=$(parse_yaml "$CONFIG_FILE" "skip_q8_baseline")
MMPROJ=$(parse_yaml "$CONFIG_FILE" "mmproj")
CALIBRATION=$(parse_yaml "$CONFIG_FILE" "calibration")
[ -z "$CALIBRATION" ] && CALIBRATION="${REPO_DIR}/calibration/calibration_v1.2.txt"

# Eval suite: CLI override > YAML > default "native"
YAML_EVAL_SUITE=$(parse_yaml "$CONFIG_FILE" "eval_suite")
[ -n "$EVAL_SUITE" ] || EVAL_SUITE="${YAML_EVAL_SUITE:-all}"

# Harness tasks (configurable per model)
HARNESS_TASKS=$(parse_yaml "$CONFIG_FILE" "harness_tasks")
[ -z "$HARNESS_TASKS" ] && HARNESS_TASKS="humaneval,mbpp,ifeval,mmlu,arc_challenge,truthfulqa_mc2"

# Native eval tasks (configurable per model)
# When eval_suite=all (default), accuracy benchmarks run via harness, not native
NATIVE_EVALS=$(parse_yaml "$CONFIG_FILE" "native_evals")
if [ -z "$NATIVE_EVALS" ]; then
    if [ "$EVAL_SUITE" = "native" ]; then
        NATIVE_EVALS="ppl,kl,hellaswag,winogrande,mmlu,arc,truthfulqa,speed"
    else
        # Default: only run PPL/KL/HellaSwag/Winogrande/speed natively
        # MMLU/ARC/TruthfulQA go through harness for proper few-shot eval
        NATIVE_EVALS="ppl,kl,hellaswag,winogrande,speed"
    fi
fi

# Load baseline downloads
mapfile -t BASELINES < <(parse_yaml_list "$CONFIG_FILE" "baselines")

# Profiles
# Mini requires imatrix (uses iq2_s), so it only runs in Phase 7 (ivariants)
PROFILES=(quality balanced compact)
I_PROFILES=(i-quality i-balanced i-compact)

echo "═══════════════════════════════════════════════════════════"
echo "  APEX Pipeline: ${MODEL_NAME}"
echo "  Model: ${MODEL_ID}"
echo "  Layers: ${LAYERS}"
echo "  HF Repo: ${HF_REPO}"
echo "  Work Dir: ${WORK_DIR}"
echo "  Calibration: ${CALIBRATION}"
echo "  Eval Suite: ${EVAL_SUITE}"
echo "  Harness Tasks: ${HARNESS_TASKS}"
echo "  Native Evals: ${NATIVE_EVALS}"
echo "═══════════════════════════════════════════════════════════"

# ─── Directory setup ───────────────────────────────────────────────────────────
MODEL_DIR="${WORK_DIR}/${CONFIG_PREFIX}"
RESULTS_DIR="${REPO_DIR}/benchmark_results/${CONFIG_PREFIX}"
CONFIGS_DIR="${REPO_DIR}/configs"
mkdir -p "$MODEL_DIR" "$RESULTS_DIR"

# ─── Phase control ─────────────────────────────────────────────────────────────
ALL_PHASES=(config download convert baseline quantize imatrix ivariants eval eval_baselines plots publish mlx)

should_run() {
    local phase="$1"
    [ -n "$PHASE" ] && { [ "$PHASE" = "$phase" ] && return 0 || return 1; }
    if [ -n "$ONLY_PHASES" ]; then
        echo "$ONLY_PHASES" | tr ',' '\n' | grep -q "^${phase}$"; return $?
    fi
    if [ -n "$SKIP_PHASES" ]; then
        if echo "$SKIP_PHASES" | tr ',' '\n' | grep -q "^${phase}$"; then return 1; fi
    fi
    if [ -n "$FROM_PHASE" ]; then
        local found=false
        for p in "${ALL_PHASES[@]}"; do
            [ "$p" = "$FROM_PHASE" ] && found=true
            $found && [ "$p" = "$phase" ] && return 0
            $found || continue
        done
        return 1
    fi
    return 0
}

should_eval() {
    local task="$1"
    echo "$NATIVE_EVALS" | tr ',' '\n' | grep -q "^${task}$"; return $?
}

# ─── Find binaries ─────────────────────────────────────────────────────────────
find_bin() {
    local name="$1"
    for d in "$LLAMA_CPP_DIR" "./llama.cpp/build/bin" "/root/llama.cpp/build/bin" "$(dirname "$0")/../llama.cpp/build/bin"; do
        [ -n "$d" ] && [ -f "$d/$name" ] && echo "$d/$name" && return 0
    done
    command -v "$name" 2>/dev/null && return 0
    echo "Error: $name not found" >&2; return 1
}

QUANTIZE=$(find_bin "llama-quantize")
PPL=$(find_bin "llama-perplexity")
BENCH=$(find_bin "llama-bench")
IMATRIX_BIN=$(find_bin "llama-imatrix")
SERVER=$(find_bin "llama-server") || true
CONVERT="$(dirname "$QUANTIZE")/../../../convert_hf_to_gguf.py"
[ -f "$CONVERT" ] || CONVERT="/root/llama.cpp/convert_hf_to_gguf.py"

# ─── Eval data paths ──────────────────────────────────────────────────────────
EVAL_DATA=""
for d in "${WORK_DIR}/eval-data" "/root/eval-data" "${HOME}/.cache/autoresearch-quant/eval-data" "${HOME}/.cache/apex-quant/eval-data"; do
    [ -d "$d" ] && EVAL_DATA="$d" && break
done
WIKITEXT=""
for d in "${WORK_DIR}/wikitext-2-raw/wiki.test.raw" "${EVAL_DATA}/wikitext-2-raw/wikitext-2-raw/wiki.test.raw" "${HOME}/.cache/autoresearch-quant/wikitext-2-raw/wiki.test.raw"; do
    [ -f "$d" ] && WIKITEXT="$d" && break
done

# ─── Helper functions ──────────────────────────────────────────────────────────

ts() { date '+%Y-%m-%d %H:%M:%S'; }
log()  { echo ""; echo "[$(ts)] ═══ $1 ═══"; echo ""; }
info() { echo "[$(ts)] $1"; }
warn() { echo "[$(ts)] ⚠ $1" >&2; }
err()  { echo "[$(ts)] ✗ $1" >&2; }

# Find the F16/BF16 source GGUF
find_f16() {
    for f in \
        "${MODEL_DIR}/f16.gguf" \
        $(find "${MODEL_DIR}" -name "*BF16*00001*.gguf" -follow 2>/dev/null) \
        $(find "${MODEL_DIR}" -name "*F16*00001*.gguf" -follow 2>/dev/null) \
        $(find "${MODEL_DIR}/source" -name "*.gguf" -follow 2>/dev/null | sort | head -1) \
        $(find "${MODEL_DIR}" -maxdepth 3 -name "*.gguf" -follow 2>/dev/null | grep -i "bf16\|f16" | sort | head -1); do
        [ -f "$f" ] && echo "$f" && return 0
    done
    echo "Error: F16/BF16 GGUF not found in ${MODEL_DIR}" >&2; return 1
}

run_native_eval() {
    local model="$1"
    local name="$2"
    local ref_logits="${3:-}"
    local outfile="${RESULTS_DIR}/${name}.json"
    local size_gb=$(echo "scale=1; $(stat -Lc%s "$model") / 1073741824" | bc)

    info "Native eval: $name ($(basename "$model"), ${size_gb} GB)"
    info "  Output: $outfile"

    local ppl="null" ppl_err="null"
    local kl_mean="null" kl_max="null" kl_99="null" kl_median="null"
    local hs="null" wg="null" mmlu="null" arc="null" tqa="null"
    local pp512="null" tg128="null"

    if should_eval ppl; then
        info "  [1/8] Perplexity (wikitext-2)..."
        local ppl_out=$($PPL -m "$model" -f "$WIKITEXT" -ngl $NGL 2>&1)
        ppl=$(echo "$ppl_out" | grep "Final estimate" | grep -oP 'PPL = \K[0-9.]+' || echo "null")
        ppl_err=$(echo "$ppl_out" | grep "Final estimate" | grep -oP '\+/- \K[0-9.]+' || echo "null")
        info "  [1/8] PPL = $ppl +/- $ppl_err"
    fi

    if should_eval kl && [ -n "$ref_logits" ] && [ -f "$ref_logits" ]; then
        info "  [2/8] KL divergence vs F16..."
        local kl_out=$($PPL -m "$model" -f "$WIKITEXT" -ngl $NGL \
            --kl-divergence --kl-divergence-base "$ref_logits" 2>&1)
        kl_mean=$(echo "$kl_out" | grep "Mean" | grep -oP 'KLD:\s+\K[0-9.]+' || echo "null")
        kl_max=$(echo "$kl_out" | grep "Maximum" | grep -oP 'KLD:\s+\K[0-9.]+' || echo "null")
        kl_99=$(echo "$kl_out" | grep "99.9%" | grep -oP 'KLD:\s+\K[0-9.]+' || echo "null")
        kl_median=$(echo "$kl_out" | grep "Median" | grep -oP 'KLD:\s+\K[0-9.]+' || echo "null")
        info "  [2/8] KL mean=$kl_mean max=$kl_max 99.9%=$kl_99"
    fi

    if should_eval hellaswag; then
        info "  [3/8] HellaSwag (400 tasks)..."
        local hs_out=$($PPL -m "$model" -f "$EVAL_DATA/hellaswag_val_full.txt" \
            --hellaswag --hellaswag-tasks 400 -c 4096 -ngl $NGL 2>&1)
        # HellaSwag outputs "N\tSCORE%\t[CI]" — last line, 2nd field, strip %
        hs=$(echo "$hs_out" | grep -P '^\d+\t' | tail -1 | awk -F'\t' '{gsub(/%/,"",$2); print $2}' || echo "null")
        [ -z "$hs" ] && hs="null"
        info "  [3/8] HellaSwag = ${hs}%"
    fi

    if should_eval winogrande; then
        info "  [4/8] Winogrande (400 tasks)..."
        local wg_out=$($PPL -m "$model" -f "$EVAL_DATA/winogrande-debiased-eval.csv" \
            --winogrande --winogrande-tasks 400 -c 4096 -ngl $NGL 2>&1)
        # Winogrande outputs "N\tSCORE\t..." — last line, 2nd field
        wg=$(echo "$wg_out" | grep -P '^\d+\t' | tail -1 | awk -F'\t' '{print $2}' || echo "null")
        [ -z "$wg" ] && wg="null"
        info "  [4/8] Winogrande = ${wg}%"
    fi

    if should_eval mmlu; then
        info "  [5/8] MMLU..."
        mmlu=$($PPL -m "$model" -bf "$EVAL_DATA/mmlu-validation.bin" \
            --multiple-choice -np 16 -c 4096 -ngl $NGL 2>&1 | grep -i "final result" | grep -oP ':\s*\K[0-9]+\.[0-9]+' | head -1 || echo "null")
        info "  [5/8] MMLU = $mmlu"
    fi

    if should_eval arc; then
        info "  [6/8] ARC-Challenge..."
        arc=$($PPL -m "$model" -bf "$EVAL_DATA/arc-challenge-validation.bin" \
            --multiple-choice -np 16 -c 4096 -ngl $NGL 2>&1 | grep -i "final result" | grep -oP ':\s*\K[0-9]+\.[0-9]+' | head -1 || echo "null")
        info "  [6/8] ARC = $arc"
    fi

    if should_eval truthfulqa; then
        info "  [7/8] TruthfulQA..."
        tqa=$($PPL -m "$model" -bf "$EVAL_DATA/truthful-qa-validation.bin" \
            --multiple-choice -np 16 -c 4096 -ngl $NGL 2>&1 | grep -i "final result" | grep -oP ':\s*\K[0-9]+\.[0-9]+' | head -1 || echo "null")
        info "  [7/8] TruthfulQA = $tqa"
    fi

    if should_eval speed; then
        info "  [8/8] Inference speed (pp512 + tg128)..."
        local speed_out=$($BENCH -m "$model" -p 512 -n 128 -ngl $NGL 2>&1)
        # llama-bench output: "| model | size | params | backend | ngl | test | t/s |"
        # The t/s value is the last float before the ± on each line
        pp512=$(echo "$speed_out" | grep "pp512" | grep -oP '[0-9]+\.[0-9]+ ±' | head -1 | grep -oP '[0-9]+\.[0-9]+' || echo "null")
        tg128=$(echo "$speed_out" | grep "tg128" | grep -oP '[0-9]+\.[0-9]+ ±' | head -1 | grep -oP '[0-9]+\.[0-9]+' || echo "null")
        info "  [8/8] pp512=${pp512} t/s, tg128=${tg128} t/s"
    fi

    # Write JSON
    cat > "$outfile" << ENDJSON
{
  "model": "$name",
  "size_gb": $size_gb,
  "perplexity": ${ppl},
  "ppl_error": ${ppl_err},
  "kl_mean": ${kl_mean},
  "kl_max": ${kl_max},
  "kl_99_9": ${kl_99},
  "kl_median": ${kl_median},
  "hellaswag": ${hs},
  "winogrande": ${wg},
  "mmlu": ${mmlu},
  "arc_challenge": ${arc},
  "truthfulqa": ${tqa},
  "pp512_ts": ${pp512},
  "tg128_ts": ${tg128}
}
ENDJSON
    info "  ✓ Results saved: $outfile"
}

run_harness_eval() {
    local model="$1"
    local name="$2"
    local outfile="${RESULTS_DIR}/${name}_harness.json"

    [ -z "$SERVER" ] && { warn "llama-server not found, skipping harness eval for $name"; return; }
    command -v lm_eval >/dev/null 2>&1 || { warn "lm_eval not installed, skipping harness eval for $name"; return; }

    info "Harness eval: $name (tasks: $HARNESS_TASKS)"
    info "  Starting llama-server on port $PORT..."
    $SERVER -m "$model" -c 8192 -ngl $NGL --port $PORT --n-predict 1024 --log-disable > /dev/null 2>&1 &
    local srv_pid=$!

    # Wait for server
    local ready=false
    for i in $(seq 1 240); do
        if curl -s "http://localhost:${PORT}/health" 2>/dev/null | grep -q "ok"; then
            info "  Server ready after ${i}s"
            ready=true
            break
        fi
        kill -0 "$srv_pid" 2>/dev/null || { warn "Server died during startup for $name"; return; }
        sleep 1
    done
    $ready || { warn "Server not ready after 240s for $name"; kill "$srv_pid" 2>/dev/null; return; }

    # Build tokenizer arg
    local tok_arg=""
    [ -n "$TOKENIZER" ] && tok_arg=",tokenizer_backend=huggingface,tokenizer=${TOKENIZER}"

    info "  Running lm_eval: $HARNESS_TASKS"
    local harness_dir=$(mktemp -d)
    lm_eval \
        --model local-completions \
        --model_args "model=${name},base_url=http://localhost:${PORT}/v1/completions${tok_arg}" \
        --tasks "$HARNESS_TASKS" \
        --batch_size 1 \
        --output_path "$harness_dir" \
        2>&1 | tail -30

    kill "$srv_pid" 2>/dev/null; wait "$srv_pid" 2>/dev/null || true
    info "  Server stopped"

    # Parse harness results into JSON
    local result_file=$(find "$harness_dir" -name "results.json" -type f 2>/dev/null | head -1)
    if [ -n "$result_file" ] && [ -f "$result_file" ]; then
        python3 << PYEOF
import json, os
with open("$result_file") as f:
    data = json.load(f)
results = data.get("results", {})
output = {"model": "$name", "eval_harness": {}}
for task, metrics in results.items():
    clean = {k: round(v, 4) if isinstance(v, float) else v
             for k, v in metrics.items() if not k.startswith("alias") and isinstance(v, (int, float))}
    if clean:
        output["eval_harness"][task] = clean
with open("$outfile", "w") as f:
    json.dump(output, f, indent=2)
print("  → $outfile")
PYEOF
    else
        warn "No harness results found for $name in $harness_dir"
    fi
    rm -rf "$harness_dir"
}

run_full_eval() {
    local model="$1"
    local name="$2"
    local ref_logits="${3:-}"

    # Native evals: PPL, KL, HellaSwag, Winogrande, speed (+ MMLU/ARC/TQA if native-only mode)
    if [ "$EVAL_SUITE" = "native" ] || [ "$EVAL_SUITE" = "all" ]; then
        run_native_eval "$model" "$name" "$ref_logits"
    fi

    # Harness evals: MMLU 5-shot, ARC 25-shot, TruthfulQA mc2, HumanEval, MBPP, IFEval
    # Default for eval_suite=all; also runs in harness-only mode
    if [ "$EVAL_SUITE" = "harness" ] || [ "$EVAL_SUITE" = "all" ]; then
        run_harness_eval "$model" "$name"
    fi
}

upload_file() {
    local file="$1"
    local fname=$(basename "$file")
    info "Uploading $(basename "$file") ($(du -h "$file" | cut -f1)) to $HF_REPO..."
    python3 -c "
from huggingface_hub import HfApi
api = HfApi()
api.upload_file(path_or_fileobj='$file', path_in_repo='$fname', repo_id='$HF_REPO')
print('  Uploaded: $fname')
"
}

# ═══════════════════════════════════════════════════════════════════════════════
# Phase 1: Config generation
# ═══════════════════════════════════════════════════════════════════════════════
if should_run config; then
    log "Phase 1: Generating configs (${LAYERS} layers, ${#PROFILES[@]} profiles)"
    for profile in "${PROFILES[@]}"; do
        "$SCRIPT_DIR/generate_config.sh" --profile "$profile" --layers "$LAYERS" \
            -o "${CONFIGS_DIR}/${CONFIG_PREFIX}_${profile}.txt"
        info "  Generated: ${CONFIG_PREFIX}_${profile}.txt"
    done
    # Also generate mini/nano/micro configs (used by ivariants phase only)
    for extra in mini nano micro; do
        "$SCRIPT_DIR/generate_config.sh" --profile "$extra" --layers "$LAYERS" \
            -o "${CONFIGS_DIR}/${CONFIG_PREFIX}_${extra}.txt"
        info "  Generated: ${CONFIG_PREFIX}_${extra}.txt (for I-${extra^})"
    done
fi

# ═══════════════════════════════════════════════════════════════════════════════
# Phase 2: Download
# ═══════════════════════════════════════════════════════════════════════════════
if should_run download; then
    log "Phase 2: Downloading models"
    export HF_HUB_ENABLE_HF_TRANSFER=1

    if [ -n "$SOURCE_GGUF" ]; then
        info "Source is pre-converted GGUF: $SOURCE_GGUF"
        mkdir -p "${MODEL_DIR}/source"
        # Only download BF16 files (not the entire repo with all quant variants)
        hf download "$SOURCE_GGUF" --include "BF16/*" --local-dir "${MODEL_DIR}/source" 2>&1 | tail -5
    else
        info "Downloading safetensors: $MODEL_ID"
        mkdir -p "${MODEL_DIR}/safetensors"
        hf download "$MODEL_ID" --local-dir "${MODEL_DIR}/safetensors" 2>&1 | tail -5
    fi

    if [ ${#BASELINES[@]} -gt 0 ]; then
        info "Downloading ${#BASELINES[@]} baselines..."
        mkdir -p "${MODEL_DIR}/baselines"
        for baseline in "${BASELINES[@]}"; do
            local_repo=$(echo "$baseline" | cut -d: -f1)
            local_pattern=$(echo "$baseline" | cut -d: -f2)
            local_label=$(echo "$baseline" | cut -d: -f3)
            info "  → $local_repo ($local_pattern) as $local_label"
            hf download "$local_repo" --include "$local_pattern" \
                --local-dir "${MODEL_DIR}/baselines/${local_label}" 2>&1 | tail -3
        done
    fi
fi

# ═══════════════════════════════════════════════════════════════════════════════
# Phase 3: Convert to F16 GGUF
# ═══════════════════════════════════════════════════════════════════════════════
if should_run convert; then
    log "Phase 3: Converting to F16 GGUF"
    F16="${MODEL_DIR}/f16.gguf"

    if [ -n "$SOURCE_GGUF" ]; then
        info "Using pre-converted GGUF source"
    elif [ -f "$F16" ]; then
        info "F16 already exists: $F16"
    else
        info "Converting safetensors → F16 GGUF..."
        python3 "$CONVERT" "${MODEL_DIR}/safetensors" --outfile "$F16" --outtype f16 2>&1 | tail -10
    fi
    F16=$(find_f16)
    ls -lh "$F16"
fi

# ═══════════════════════════════════════════════════════════════════════════════
# Phase 4: Baselines (F16 + Q8_0)
# ═══════════════════════════════════════════════════════════════════════════════
if should_run baseline; then
    log "Phase 4: Running baselines"
    F16=$(find_f16)

    info "Generating F16 reference logits (this may take a while for large models)..."
    $PPL -m "$F16" -f "$WIKITEXT" -ngl $NGL \
        --save-all-logits "${MODEL_DIR}/reference-logits.bin" > "${MODEL_DIR}/f16_ppl.log" 2>&1 || true
    grep "Final" "${MODEL_DIR}/f16_ppl.log" | tail -1 || true
    info "Reference logits saved: $(ls -lh "${MODEL_DIR}/reference-logits.bin" 2>/dev/null | awk '{print $5}')"

    info "F16 speed benchmark..."
    $BENCH -m "$F16" -p 512 -n 128 -ngl $NGL 2>&1 | grep -E "pp512|tg128" || true

    # Q8_0 baseline: use pre-made from baselines (downloaded in Phase 2), not self-quantized
    q8_gguf=$(find "${MODEL_DIR}/baselines" -follow -name "*Q8_0*" -name "*.gguf" 2>/dev/null \
        | awk '/-[0-9]+-of-/{if(/-00001-of-/)print; next} {print}' | head -1)
    if [ -n "$q8_gguf" ] && [ -f "$q8_gguf" ] && [ "$SKIP_Q8_BASELINE" != "true" ]; then
        info "Evaluating Q8_0 baseline: $(basename "$q8_gguf")"
        run_full_eval "$q8_gguf" "q8_0" "${MODEL_DIR}/reference-logits.bin"
    elif [ "$SKIP_Q8_BASELINE" != "true" ]; then
        warn "No Q8_0 baseline found in baselines/ — add it to the YAML baselines list"
    fi
fi

# ═══════════════════════════════════════════════════════════════════════════════
# Phase 5: APEX quantization (standard profiles)
# ═══════════════════════════════════════════════════════════════════════════════
if should_run quantize; then
    log "Phase 5: APEX quantization"
    F16=$(find_f16)

    declare -A BASE_TYPES=([quality]=Q6_K [balanced]=Q5_K [compact]=Q4_K [mini]=Q3_K)

    for profile in "${PROFILES[@]}"; do
        cap="$(echo ${profile:0:1} | tr a-z A-Z)${profile:1}"
        outfile="${MODEL_DIR}/${MODEL_NAME}-APEX-${cap}.gguf"
        config="${CONFIGS_DIR}/${CONFIG_PREFIX}_${profile}.txt"
        base="${BASE_TYPES[$profile]}"

        info "Quantizing APEX-${cap} (base: ${base}, config: ${config})..."
        $QUANTIZE --tensor-type-file "$config" "$F16" "$outfile" "$base" 2>&1 | tail -3
        ls -lh "$outfile"
    done
fi

# ═══════════════════════════════════════════════════════════════════════════════
# Phase 6: Generate imatrix
# ═══════════════════════════════════════════════════════════════════════════════
if should_run imatrix; then
    log "Phase 6: Generating importance matrix"
    F16=$(find_f16)

    info "Calibration file: $CALIBRATION ($(wc -l < "$CALIBRATION") lines, $(du -h "$CALIBRATION" | cut -f1))"
    info "Source model: $F16"
    $IMATRIX_BIN -m "$F16" -f "$CALIBRATION" -ngl $NGL \
        -o "${MODEL_DIR}/imatrix.dat" 2>&1 | tail -5
    ls -lh "${MODEL_DIR}/imatrix.dat"
fi

# ═══════════════════════════════════════════════════════════════════════════════
# Phase 7: I-variants (with imatrix) — includes I-Mini
# ═══════════════════════════════════════════════════════════════════════════════
if should_run ivariants; then
    log "Phase 7: I-variant quantization (with imatrix)"
    F16=$(find_f16)
    IMAT="${MODEL_DIR}/imatrix.dat"
    [ -f "$IMAT" ] || { err "imatrix.dat not found at $IMAT — run imatrix phase first"; exit 1; }

    declare -A I_BASE_TYPES=([i-quality]=Q6_K [i-balanced]=Q5_K [i-compact]=Q4_K [i-mini]=Q3_K [i-nano]=iq2_xxs [i-micro]=iq1_m)

    for profile in "${I_PROFILES[@]}" i-mini i-nano i-micro; do
        config_name="${profile#i-}"
        cap="I-$(echo ${config_name:0:1} | tr a-z A-Z)${config_name:1}"
        outfile="${MODEL_DIR}/${MODEL_NAME}-APEX-${cap}.gguf"
        config="${CONFIGS_DIR}/${CONFIG_PREFIX}_${config_name}.txt"
        base="${I_BASE_TYPES[$profile]}"

        info "Quantizing APEX-${cap} (base: ${base} + imatrix)..."
        $QUANTIZE --tensor-type-file "$config" --imatrix "$IMAT" \
            "$F16" "$outfile" "$base" 2>&1 | tail -3
        ls -lh "$outfile"
    done
fi

# ═══════════════════════════════════════════════════════════════════════════════
# Phase 8: Evaluate all APEX quants
# ═══════════════════════════════════════════════════════════════════════════════
if should_run eval; then
    log "Phase 8: Evaluating APEX quants"
    REF_LOGITS="${MODEL_DIR}/reference-logits.bin"

    for gguf in "${MODEL_DIR}/${MODEL_NAME}-APEX-"*.gguf; do
        [ -f "$gguf" ] || continue
        fname=$(basename "$gguf" .gguf)
        name=$(echo "$fname" | sed "s/${MODEL_NAME}-//" | tr '[:upper:]' '[:lower:]' | tr '-' '_')
        echo ""
        echo "  ── $name ──"
        run_full_eval "$gguf" "$name" "$REF_LOGITS"
    done
fi

# ═══════════════════════════════════════════════════════════════════════════════
# Phase 9: Evaluate baselines
# ═══════════════════════════════════════════════════════════════════════════════
if should_run eval_baselines; then
    log "Phase 9: Evaluating baselines"
    REF_LOGITS="${MODEL_DIR}/reference-logits.bin"

    # Find single GGUFs (no -of- pattern) or first split (-00001-of-)
    for gguf in $(find "${MODEL_DIR}/baselines" -follow -name "*.gguf" 2>/dev/null \
        | awk '/-[0-9]+-of-/{if(/-00001-of-/)print; next} {print}' | sort -u); do
        [ -f "$gguf" ] || continue
        name="baseline_$(basename "$gguf" .gguf | tr '[:upper:]' '[:lower:]' | tr '.-' '_')"
        echo ""
        echo "  ── $name ──"
        run_full_eval "$gguf" "$name" "$REF_LOGITS"
    done
fi

# ═══════════════════════════════════════════════════════════════════════════════
# Phase 10: Generate plots
# ═══════════════════════════════════════════════════════════════════════════════
if should_run plots; then
    log "Phase 10: Generating plots"
    PLOTS_DIR="${RESULTS_DIR}/plots"
    mkdir -p "$PLOTS_DIR"
    python3 "$SCRIPT_DIR/plot_benchmarks.py" \
        --input-dir "$RESULTS_DIR" \
        --output-dir "$PLOTS_DIR" 2>&1
fi

# ═══════════════════════════════════════════════════════════════════════════════
# Phase 11: Publish to HuggingFace
# ═══════════════════════════════════════════════════════════════════════════════
if should_run publish; then
    log "Phase 11: Publishing to HuggingFace"

    python3 -c "
from huggingface_hub import create_repo
create_repo('${HF_REPO}', repo_type='model', exist_ok=True)
print('Repo ready: ${HF_REPO}')
"
    for gguf in "${MODEL_DIR}/${MODEL_NAME}-APEX-"*.gguf; do
        [ -f "$gguf" ] || continue
        upload_file "$gguf"
    done

    # Upload mmproj if configured (vision models)
    if [ -n "$MMPROJ" ]; then
        mmproj_file="${MODEL_DIR}/mmproj.gguf"
        if [ -f "$mmproj_file" ]; then
            info "Uploading mmproj (vision projection)..."
            upload_file "$mmproj_file"
        else
            info "mmproj configured but file not found at $mmproj_file, downloading..."
            mmproj_repo="${MMPROJ%%:*}"
            mmproj_name="${MMPROJ##*:}"
            hf download "$mmproj_repo" "$mmproj_name" --local-dir "${MODEL_DIR}/mmproj_dl" 2>/dev/null
            dl_file=$(find "${MODEL_DIR}/mmproj_dl" -name "$mmproj_name" -type f 2>/dev/null | head -1)
            if [ -n "$dl_file" ] && [ -f "$dl_file" ]; then
                cp "$dl_file" "$mmproj_file"
                upload_file "$mmproj_file"
                rm -rf "${MODEL_DIR}/mmproj_dl"
            else
                warn "Could not download mmproj from $MMPROJ"
            fi
        fi
    fi

    # Upload model card (README.md) if available
    modelcard="${REPO_DIR}/model_cards/${CONFIG_PREFIX}_modelcard.md"
    if [ -f "$modelcard" ]; then
        info "Uploading model card..."
        python3 -c "
from huggingface_hub import HfApi
api = HfApi()
api.upload_file(path_or_fileobj='${modelcard}', path_in_repo='README.md', repo_id='${HF_REPO}')
print('  Model card uploaded')
"
    fi

    echo "  Published: https://huggingface.co/${HF_REPO}"
fi

# ═══════════════════════════════════════════════════════════════════════════════
# Phase 12: MLX conversion (optional)
# ═══════════════════════════════════════════════════════════════════════════════
if should_run mlx; then
    log "Phase 12: MLX conversion"

    # Check if mlx-lm is available
    if ! python3 -c "from mlx_lm import convert" 2>/dev/null; then
        warn "mlx-lm not installed, skipping MLX conversion"
        warn "Install with: pip install 'mlx[cuda13]' mlx-lm"
    else
        # Set LD_LIBRARY_PATH for CUDA 13 libs if needed
        export LD_LIBRARY_PATH="/usr/local/lib/python3.10/dist-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"

        MLX_CONVERTER="${REPO_DIR}/scripts/convert_to_mlx.py"
        MLX_DIR="${MODEL_DIR}/mlx"
        mkdir -p "$MLX_DIR"

        # Determine the source model ID for mlx_lm (original HF weights)
        MLX_SOURCE="${MODEL_ID}"

        for profile in quality balanced compact; do
            apex_config="${CONFIGS_DIR}/${CONFIG_PREFIX}_${profile}.txt"
            [ -f "$apex_config" ] || continue

            mlx_repo="${HF_REPO%-GGUF}-APEX-${profile^}-MLX"
            mlx_outdir="${MLX_DIR}/${profile}"

            info "MLX ${profile^}: converting from ${MLX_SOURCE}..."

            # Generate MLX config
            mlx_config="${MLX_DIR}/${CONFIG_PREFIX}_${profile}_mlx.json"
            python3 "$MLX_CONVERTER" --apex-config "$apex_config" --output "$mlx_config" 2>&1 | tail -3

            # Run conversion
            if python3 -c "
import json
from mlx_lm import convert

with open('${mlx_config}') as f:
    qconfig = json.load(f)

convert(
    model='${MLX_SOURCE}',
    quantize=True,
    q_bits=qconfig.get('bits', 4),
    q_group_size=qconfig.get('group_size', 64),
    q_predicate=None,
    mlx_path='${mlx_outdir}',
)
print('Conversion complete')
" 2>&1 | tail -5; then
                info "  MLX ${profile^} converted to ${mlx_outdir}"

                # Upload to HF
                python3 -c "
from huggingface_hub import HfApi, create_repo
create_repo('${mlx_repo}', repo_type='model', exist_ok=True)
api = HfApi()
api.upload_folder(folder_path='${mlx_outdir}', repo_id='${mlx_repo}', repo_type='model')
print('  Uploaded: ${mlx_repo}')
" 2>&1 | tail -3
            else
                warn "MLX ${profile^} conversion failed"
            fi
        done

        # Clean up MLX temp files
        rm -rf "$MLX_DIR"
        info "MLX conversion complete"
    fi
fi

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  Pipeline complete: ${MODEL_NAME}"
echo "  Results: ${RESULTS_DIR}/"
echo "═══════════════════════════════════════════════════════════"
