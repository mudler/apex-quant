#!/usr/bin/env bash
#
# generate_config.sh — Generate APEX tensor-type configuration files
#
# Creates a tensor-type file for llama-quantize's --tensor-type-file flag.
# Supports any number of layers and all APEX profiles.
#
# Usage:
#   ./scripts/generate_config.sh --profile balanced --layers 40 > config.txt
#   ./scripts/generate_config.sh --profile mini --layers 40 -o configs/my_config.txt
#   ./scripts/generate_config.sh --custom --edge-exp Q6_K --mid-exp Q4_K \
#       --shared Q8_0 --attn Q6_K --layers 40 > config.txt
#
# Profiles:
#   quality     Q6_K/Q5_K/IQ4_XS experts, Q8_0 shared, Q6_K attn
#   i-quality   Same as quality (use with --imatrix at quantize time)
#   balanced    Q6_K/Q5_K experts, Q8_0 shared, Q6_K attn
#   i-balanced  Same as balanced (use with --imatrix at quantize time)
#   compact     Q4_K/Q3_K experts, Q6_K shared, Q4_K attn
#   i-compact   Same as compact (use with --imatrix at quantize time)
#   mini        Q3_K edge / IQ2_S mid experts, Q5_K/Q4_K shared, Q4_K/Q3_K attn
#   nano        Q3_K edge / IQ2_S near / IQ2_XXS mid experts (2.06 bpw mid) — needs imatrix
#   micro       Q3_K edge / IQ2_XS near / IQ1_M mid experts (1.75 bpw mid) — needs imatrix, experimental
#   custom      Specify each type manually via flags
#
set -euo pipefail

# Defaults
PROFILE=""
LAYERS=40
OUTPUT=""
# Custom mode overrides
EDGE_EXP="" NEAR_EXP="" MID_EXP=""
EDGE_SHARED="" MID_SHARED=""
EDGE_ATTN="" MID_ATTN=""

show_help() {
    sed -n '3,25p' "$0"
    exit 0
}

while [ $# -gt 0 ]; do
    case "$1" in
        --profile|-p)      PROFILE="$2"; shift 2 ;;
        --layers|-l)       LAYERS="$2"; shift 2 ;;
        --output|-o)       OUTPUT="$2"; shift 2 ;;
        --custom)          PROFILE="custom"; shift ;;
        --edge-exp)        EDGE_EXP="$2"; shift 2 ;;
        --near-exp)        NEAR_EXP="$2"; shift 2 ;;
        --mid-exp)         MID_EXP="$2"; shift 2 ;;
        --edge-shared)     EDGE_SHARED="$2"; shift 2 ;;
        --mid-shared)      MID_SHARED="$2"; shift 2 ;;
        --edge-attn)       EDGE_ATTN="$2"; shift 2 ;;
        --mid-attn)        MID_ATTN="$2"; shift 2 ;;
        --help|-h)         show_help ;;
        *)                 echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

[ -z "$PROFILE" ] && { echo "Error: --profile required" >&2; exit 1; }

# Edge boundaries (first/last N layers get higher precision)
EDGE_HI=4                         # L0..EDGE_HI
EDGE_LO=$(( LAYERS - 5 ))         # EDGE_LO..LAYERS-1
NEAR_HI=9                         # EDGE_HI+1..NEAR_HI
NEAR_LO=$(( LAYERS - 10 ))        # NEAR_LO..EDGE_LO-1

# Set types per profile
case "$PROFILE" in
    quality|i-quality)
        EDGE_EXP="${EDGE_EXP:-Q6_K}"
        NEAR_EXP="${NEAR_EXP:-Q5_K}"
        MID_EXP="${MID_EXP:-iq4_xs}"
        EDGE_SHARED="${EDGE_SHARED:-Q8_0}"
        MID_SHARED="${MID_SHARED:-Q8_0}"
        EDGE_ATTN="${EDGE_ATTN:-Q6_K}"
        MID_ATTN="${MID_ATTN:-Q6_K}"
        ;;
    balanced|i-balanced)
        EDGE_EXP="${EDGE_EXP:-Q6_K}"
        NEAR_EXP="${NEAR_EXP:-Q5_K}"
        MID_EXP="${MID_EXP:-Q5_K}"
        EDGE_SHARED="${EDGE_SHARED:-Q8_0}"
        MID_SHARED="${MID_SHARED:-Q8_0}"
        EDGE_ATTN="${EDGE_ATTN:-Q6_K}"
        MID_ATTN="${MID_ATTN:-Q6_K}"
        ;;
    compact|i-compact)
        EDGE_EXP="${EDGE_EXP:-Q4_K}"
        NEAR_EXP="${NEAR_EXP:-Q3_K}"
        MID_EXP="${MID_EXP:-Q3_K}"
        EDGE_SHARED="${EDGE_SHARED:-Q6_K}"
        MID_SHARED="${MID_SHARED:-Q6_K}"
        EDGE_ATTN="${EDGE_ATTN:-Q4_K}"
        MID_ATTN="${MID_ATTN:-Q4_K}"
        ;;
    mini)
        EDGE_EXP="${EDGE_EXP:-Q3_K}"
        NEAR_EXP="${NEAR_EXP:-Q3_K}"
        MID_EXP="${MID_EXP:-iq2_s}"
        EDGE_SHARED="${EDGE_SHARED:-Q5_K}"
        MID_SHARED="${MID_SHARED:-Q4_K}"
        EDGE_ATTN="${EDGE_ATTN:-Q4_K}"
        MID_ATTN="${MID_ATTN:-Q3_K}"
        ;;
    nano|i-nano)
        # APEX Nano — aggressive mid-layer routed experts at IQ2_XXS (2.06 bpw)
        # Target: ~25-30% smaller than Mini at modest quality cost. Requires imatrix.
        EDGE_EXP="${EDGE_EXP:-Q3_K}"
        NEAR_EXP="${NEAR_EXP:-iq2_s}"
        MID_EXP="${MID_EXP:-iq2_xxs}"
        EDGE_SHARED="${EDGE_SHARED:-Q5_K}"
        MID_SHARED="${MID_SHARED:-Q4_K}"
        EDGE_ATTN="${EDGE_ATTN:-Q4_K}"
        MID_ATTN="${MID_ATTN:-Q3_K}"
        ;;
    micro|i-micro)
        # APEX Micro — extreme mid-layer routed experts at IQ1_M (1.75 bpw)
        # Only viable on MoE: sparse expert activation + shared expert kept high-precision
        # softens per-token error. Quality drop expected — experimental tier. Requires imatrix.
        EDGE_EXP="${EDGE_EXP:-Q3_K}"
        NEAR_EXP="${NEAR_EXP:-iq2_xs}"
        MID_EXP="${MID_EXP:-iq1_m}"
        EDGE_SHARED="${EDGE_SHARED:-Q5_K}"
        MID_SHARED="${MID_SHARED:-Q4_K}"
        EDGE_ATTN="${EDGE_ATTN:-Q4_K}"
        MID_ATTN="${MID_ATTN:-Q3_K}"
        ;;
    custom)
        [ -z "$EDGE_EXP" ] && { echo "Error: --custom requires --edge-exp" >&2; exit 1; }
        [ -z "$MID_EXP" ] && MID_EXP="$EDGE_EXP"
        [ -z "$NEAR_EXP" ] && NEAR_EXP="$EDGE_EXP"
        [ -z "$EDGE_SHARED" ] && EDGE_SHARED="Q8_0"
        [ -z "$MID_SHARED" ] && MID_SHARED="$EDGE_SHARED"
        [ -z "$EDGE_ATTN" ] && EDGE_ATTN="Q6_K"
        [ -z "$MID_ATTN" ] && MID_ATTN="$EDGE_ATTN"
        ;;
    *)
        echo "Error: unknown profile '$PROFILE'" >&2
        echo "Available: quality, i-quality, balanced, i-balanced, compact, i-compact, mini, nano, i-nano, micro, i-micro, custom" >&2
        exit 1
        ;;
esac

# Generate config
generate() {
    for (( i=0; i<LAYERS; i++ )); do
        # Expert type based on layer position
        if (( i <= EDGE_HI || i >= EDGE_LO )); then
            exp_type="$EDGE_EXP"
        elif (( i <= NEAR_HI || i >= NEAR_LO )); then
            exp_type="$NEAR_EXP"
        else
            exp_type="$MID_EXP"
        fi

        # Shared type based on layer position
        if (( i <= EDGE_HI || i >= EDGE_LO )); then
            shared_type="$EDGE_SHARED"
        else
            shared_type="$MID_SHARED"
        fi

        # Attention type based on layer position
        if (( i <= 2 || i >= LAYERS - 3 )); then
            attn_type="$EDGE_ATTN"
        else
            attn_type="$MID_ATTN"
        fi

        # Expert tensors
        echo "blk.${i}.ffn_gate_exps=${exp_type}"
        echo "blk.${i}.ffn_up_exps=${exp_type}"
        echo "blk.${i}.ffn_down_exps=${exp_type}"

        # Shared expert tensors
        echo "blk.${i}.ffn_gate_shexp=${shared_type}"
        echo "blk.${i}.ffn_up_shexp=${shared_type}"
        echo "blk.${i}.ffn_down_shexp=${shared_type}"

        # Attention tensors
        echo "blk.${i}.attn_q=${attn_type}"
        echo "blk.${i}.attn_k=${attn_type}"
        echo "blk.${i}.attn_v=${attn_type}"
        echo "blk.${i}.attn_output=${attn_type}"
        echo "blk.${i}.attn_gate=${attn_type}"
        echo "blk.${i}.attn_qkv=${attn_type}"

        # SSM tensors
        echo "blk.${i}.ssm_alpha=${attn_type}"
        echo "blk.${i}.ssm_beta=${attn_type}"
        echo "blk.${i}.ssm_out=${attn_type}"
    done
}

if [ -n "$OUTPUT" ]; then
    generate > "$OUTPUT"
    echo "Config written to: $OUTPUT ($(wc -l < "$OUTPUT") lines, $LAYERS layers)" >&2
else
    generate
fi
