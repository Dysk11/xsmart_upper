#!/usr/bin/env bash
set -euo pipefail

VIDEO_PATH="${1:-outputs/video/record_20260708_135111.mp4}"
OUTPUT_ROOT="${2:-outputs/benchmarks/ai_transport_ab}"
FPS="${FPS:-60}"
WARMUP_SEC="${WARMUP_SEC:-10}"
DURATION_SEC="${DURATION_SEC:-60}"
PRODUCER_DURATION_SEC="${PRODUCER_DURATION_SEC:-80}"
SHM_NAME="${SHM_NAME:-xsmart_ai_transport_bench_$$}"

mkdir -p "${OUTPUT_ROOT}"
orders=("rgb_bgr_copy rgb_lease" "rgb_lease rgb_bgr_copy" "rgb_bgr_copy rgb_lease")
start_frame_id=3000000

for trial_index in 1 2 3; do
  for transport in ${orders[$((trial_index - 1))]}; do
    run_dir="${OUTPUT_ROOT}/${transport}_run${trial_index}"
    mkdir -p "${run_dir}"
    python3 tools/replay_video_to_shm.py \
      --video "${VIDEO_PATH}" \
      --name "${SHM_NAME}" \
      --fps "${FPS}" \
      --duration-sec "${PRODUCER_DURATION_SEC}" \
      --start-frame-id "${start_frame_id}" \
      > "${run_dir}/producer.log" 2>&1 &
    producer_pid=$!
    sleep 1
    set +e
    python3 tools/benchmark_latency.py \
      --mode shared_memory \
      --shared-memory-name "${SHM_NAME}" \
      --bridge mock \
      --object-backend c_api \
      --ai-frame-transport "${transport}" \
      --warmup-sec "${WARMUP_SEC}" \
      --duration-sec "${DURATION_SEC}" \
      --output-dir "${run_dir}" \
      > "${run_dir}/benchmark.log" 2>&1
    benchmark_status=$?
    set -e
    wait "${producer_pid}"
    if [[ "${benchmark_status}" -ne 0 ]]; then
      exit "${benchmark_status}"
    fi
    start_frame_id=$((start_frame_id + 100000))
  done
done

python3 tools/summarize_ai_transport_ab.py \
  "${OUTPUT_ROOT}" \
  --output "${OUTPUT_ROOT}/summary.json"
printf 'complete\n' > "${OUTPUT_ROOT}/COMPLETE"
