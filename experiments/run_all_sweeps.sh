#!/usr/bin/env bash
# run_all_sweeps.sh — esegue tutti gli sweep in sequenza
# Esegui da repo root (tests/fixedness_test/): bash experiments/run_all_sweeps.sh
# Opzioni:
#   --skip <nome>   salta uno sweep (es. --skip timing)
#   --only <nome>   esegue solo quello sweep

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON="$REPO_ROOT/.venv/bin/python"
SKIP=()
ONLY=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip) SKIP+=("$2"); shift 2 ;;
        --only) ONLY="$2"; shift 2 ;;
        *) echo "Opzione sconosciuta: $1"; exit 1 ;;
    esac
done

should_run() {
    local name="$1"
    [[ -n "$ONLY" ]] && [[ "$ONLY" != "$name" ]] && return 1
    for s in "${SKIP[@]}"; do [[ "$s" == "$name" ]] && return 1; done
    return 0
}

run_sweep() {
    local name="$1"
    local script="$2"
    should_run "$name" || { echo "⏭  Salto: $name"; return; }
    echo ""
    echo "══════════════════════════════════════════════════════"
    echo "  ▶  $name"
    echo "══════════════════════════════════════════════════════"
    local t0=$SECONDS
    "$PYTHON" "$script"
    local elapsed=$(( SECONDS - t0 ))
    printf "  ✓  %s completato in %dm%02ds\n" "$name" $(( elapsed/60 )) $(( elapsed%60 ))
}

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║         Fixedness Sweep Suite                        ║"
echo "╚══════════════════════════════════════════════════════╝"
echo "  Python: $PYTHON"
echo "  Data:   $(date '+%Y-%m-%d %H:%M:%S')"
[[ ${#SKIP[@]} -gt 0 ]] && echo "  Skip:   ${SKIP[*]}"
[[ -n "$ONLY" ]]         && echo "  Only:   $ONLY"

T_START=$SECONDS

cd "$REPO_ROOT"

run_sweep "rho"           "$SCRIPT_DIR/sweep_rho.py"
run_sweep "rho_qi"        "$SCRIPT_DIR/sweep_rho_qi.py"
run_sweep "crossproduct"  "$SCRIPT_DIR/sweep_crossproduct.py"
run_sweep "multiseed"     "$SCRIPT_DIR/sweep_multiseed.py"
run_sweep "oracle"        "$SCRIPT_DIR/sweep_oracle.py"
run_sweep "qi"            "$SCRIPT_DIR/sweep_qi.py"
run_sweep "bksa"          "$SCRIPT_DIR/sweep_bksa.py"
run_sweep "timing"        "$SCRIPT_DIR/sweep_timing.py"

TOTAL=$(( SECONDS - T_START ))
echo ""
echo "══════════════════════════════════════════════════════"
printf "  Totale: %dm%02ds\n" $(( TOTAL/60 )) $(( TOTAL%60 ))
echo "  Dashboard: $PYTHON $SCRIPT_DIR/dashboard.py"
echo "══════════════════════════════════════════════════════"
echo ""
