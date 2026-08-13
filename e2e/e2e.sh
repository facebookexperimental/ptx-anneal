#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# e2e.sh - ptx-anneal end-to-end validation.
#
#   MODE=tune (default): run the FACTORY only -- search -> score -> admit an ACF for a captured task
#   (../sample_tasks/<name>) with CompileIQ and ptxas >= MIN_PTXAS (the GA `--apply-controls` path).
#   No frontend, no collect/consume. Pure Python + ptxas + the engine: this mode is the portable one
#   and MUST keep working in a plain git checkout with no build system.
#
#   MODE=full: the whole 3-step flow -- collect -> factory -> consume -- on a registered kernel.
#   This needs a *frontend*: a triton built with the magnon collector/consumer. Building one is
#   site-specific (in fbcode it is a buck target), so the entire frontend half lives in the optional
#   hook fb/e2e_internal.sh, sourced below when present. Without the hook, MODE=full is unavailable
#   and MODE=tune still works.
#
# One toolchain: CompileIQ + ptxas >= MIN_PTXAS. `ptxas --apply-controls` is GA only from there on,
# and any frontend must run the SAME ptxas as the factory (differing ptxas => different PTX ISA
# version => different store key => silent MISS), so there is nothing to select between.
#
# This harness is dev/test only and is never packaged (see README.md).
#
# Usage:   e2e.sh                              # MODE=tune on ../sample_tasks/sample_task
#          TASK=../sample_tasks/<name> e2e.sh  # a different captured task
#          MODE=full e2e.sh [kernel ...]       # 3-step collect->factory->consume (needs the hook)
# Env [default]: GPU[0] BENCH[cudagraph] FORCE_ADMIT[1] PTXAS[PATH] SS[$COMPILE_IQ_SEARCH_SPACE_BIN]
#   PYTHON[python3] TASK STORE   (the hook may add its own; see fb/e2e_internal.sh)
set -euo pipefail

MODE="${MODE:-tune}"
GPU="${GPU:-0}"
WORK="${WORK:-/tmp/ptx_anneal_e2e}"
BENCH="${BENCH:-cudagraph}"
FORCE_ADMIT="${FORCE_ADMIT:-1}"
PYTHON="${PYTHON:-python3}"
CIQ_POOL="${CIQ_POOL:-8}"; CIQ_GENERATIONS="${CIQ_GENERATIONS:-1}"; CIQ_CULL="${CIQ_CULL:-4}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FBCODE="$(cd "$HERE/../../../.." 2>/dev/null && pwd || true)"
HARNESS_PY="${HARNESS_PY:-$PYTHON}"

# The ptxas floor has ONE source of truth: the package enforces it, this script only reports/uses it.
# Hardcoding it here too would be the very failure this harness exists to catch -- two components
# disagreeing about a ptxas version. The literal is a fallback for when ptx_anneal is not importable
# yet (uninstalled checkout); the factory would fail on its own gate later anyway.
MIN_PTXAS="$("$HARNESS_PY" -c 'from ptx_anneal.provision import MIN_PTXAS; print(MIN_PTXAS)' 2>/dev/null || echo 13.3)"

# Optional site hook -- absent in a plain git checkout (it is ShipIt-stripped). It may set toolchain
# defaults (e.g. an in-repo ptxas) and define run_full() to provide MODE=full. Sourced before the
# defaults below so anything it sets wins over them but still loses to the caller's environment.
_HOOK="${FBCODE:+$FBCODE/triton/tools/ptx_anneal/fb/e2e_internal.sh}"
if [[ -n "${_HOOK:-}" && -f "$_HOOK" ]]; then
  # shellcheck source=/dev/null
  source "$_HOOK"
fi

PTXAS="${PTXAS:-ptxas}"
SS="${SS:-${COMPILE_IQ_SEARCH_SPACE_BIN:-}}"
ENGINE_PY="${ENGINE_PY:-$HARNESS_PY}"

if [[ -z "$SS" ]]; then
  echo "e2e: no engine search space -- set SS=<path> or \$COMPILE_IQ_SEARCH_SPACE_BIN" >&2; exit 2
fi

# Run the factory (ptx_anneal.cli) on one task: search -> score -> admit into <store_dir>. Shared by
# both modes. Returns 0 iff an ACF was admitted.
run_factory() {  # <task_dir> <store_dir> <logfile>
  local task="$1" store="$2" log="$3"
  CUDA_VISIBLE_DEVICES="$GPU" \
    env PTXAS="$PTXAS" TRITON_PTXAS_BLACKWELL_PATH="$PTXAS" PTX_ANNEAL_ENGINE_PYTHON="$ENGINE_PY" \
      PTX_ANNEAL_PTXAS_TIMEOUT="${PTX_ANNEAL_PTXAS_TIMEOUT:-120}" PTX_ANNEAL_FORCE_ADMIT="$FORCE_ADMIT" \
      CIQ_POOL="$CIQ_POOL" CIQ_GENERATIONS="$CIQ_GENERATIONS" CIQ_CULL="$CIQ_CULL" \
      "$HARNESS_PY" -m ptx_anneal.cli --ptxas "$PTXAS" --ss "$SS" \
        --task "$task" --output "$store" --bench "$BENCH" 2>&1 | tee "$log"
  grep -q "admitted ACF" "$log"
}

if [[ "$MODE" == "tune" ]]; then
  TASK="${TASK:-$HERE/../sample_tasks/sample_task}"
  store="${STORE:-$WORK/tune}"; mkdir -p "$store"
  echo "== [tune] ptx-anneal FACTORY ONLY (ptxas>=$MIN_PTXAS) on $TASK -- search+score+admit; NO collect/consume =="
  if run_factory "$TASK" "$store" "$WORK/tune.log"; then
    echo "e2e: PASS(tune) - FACTORY-only: searched+admitted+stored ACF (no consume): $(grep 'admitted ACF' "$WORK/tune.log" | tail -1)"; exit 0
  fi
  echo "e2e: FAIL(tune) - FACTORY-only: no ACF admitted (all candidates invalid)"; exit 1
fi

# --- MODE=full: provided by the site hook ---------------------------------------------------------
# The frontend (a triton with the magnon collector/consumer) is built site-specifically, so run_full
# lives in fb/e2e_internal.sh. TODO: offer a build-system-agnostic frontend so internal users working
# from a git checkout (no buck) can run MODE=full too -- today they are limited to MODE=tune.
if [[ "$MODE" == "full" ]]; then
  if declare -f run_full >/dev/null 2>&1; then run_full "$@"; exit $?; fi
  echo "e2e: MODE=full needs a frontend (a triton with the magnon collector/consumer), which is" >&2
  echo "     provided by the site hook fb/e2e_internal.sh -- not available in this checkout." >&2
  echo "     MODE=tune (the factory) works here." >&2
  exit 2
fi

echo "e2e: unknown MODE=$MODE (want tune|full)" >&2; exit 2
