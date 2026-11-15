#!/usr/bin/env bash
# Phase 2 of the v1.0 release: rebuild features once, then every grouped-CV experiment (E05-E09)
# against the v1 candidate. Resume-safe: a run whose reports/cv_results_<tag>.json exists is skipped.
#   bash scripts/run_release_experiments.sh 2>&1 | tee reports/release_run.log
set -e
cd "$(dirname "$0")/.."
PY=${PY:-python}
F=features_physics.parquet
if [ "${REBUILD:-1}" = "1" ]; then
  echo "== build features $(date +%T)"
  $PY -u scripts/02_build_features.py 2>&1 | grep -v "Parallel(n_jobs\|Done " | tail -3
fi
run() {  # run <tag> <args...>
  tag=$1; shift
  if [ -f "reports/cv_results_${tag}.json" ] && [ "${FORCE:-0}" != "1" ]; then echo "== skip $tag (done)"; return; fi
  echo "== $tag $(date +%T)"
  $PY -u scripts/06_cross_validation.py --schemes grouped --tag "$tag" "$@" 2>&1 \
    | grep -E --line-buffered "features \||fold [0-9]:|macro_f1 |pooled|Traceback|Error"
}
V1_DROP="PHY- PHYDRIFT- DRIFT0- OSC120- OSCN- MASK-"
run v1   --recommended                                         # v1 candidate (= E04 osc + live drift), fold models saved
$PY -u scripts/03_train_baseline.py --all-real | tail -3        # deployable model (all wells)
run e05  --features $F --drop-sensors DRIFT0- OSC120- OSCN- MASK- --drop-features "PHY-*__mean" "PHY-*__last" --no-save-models
run e06  --features $F --drop-sensors PHY- PHYDRIFT- DRIFT0- OSC- OSCN- MASK- --no-save-models
run e07  --features $F --drop-sensors PHY- PHYDRIFT- DRIFT0- OSC120- MASK- --no-save-models
run e08a --features $F --drop-sensors PHY- PHYDRIFT- DRIFT0- OSC120- OSCN- --no-save-models
run e08b --recommended --min-real-availability 0.7 --no-save-models
run e08c --recommended --drop-sensors T-MON-CKP --no-save-models
run e09a --recommended --no-simulated --no-drawn --no-save-models
run e09c --recommended --sim-rescale --no-save-models
echo "== ALL DONE $(date +%T)"
