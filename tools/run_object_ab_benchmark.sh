#!/usr/bin/env bash
set -euo pipefail

VIDEO_PATH="${1:-outputs/video/record_20260708_135111.mp4}"
OUTPUT_ROOT="${2:-outputs/benchmarks/object_capi_ab}"
FPS="${FPS:-60}"
WARMUP_SEC="${WARMUP_SEC:-10}"
DURATION_SEC="${DURATION_SEC:-60}"
PRODUCER_DURATION_SEC="${PRODUCER_DURATION_SEC:-75}"

mkdir -p "${OUTPUT_ROOT}"
orders=("lite2 c_api" "c_api lite2" "lite2 c_api")
start_frame_id=2000000

for trial_index in 1 2 3; do
  for backend in ${orders[$((trial_index - 1))]}; do
    run_dir="${OUTPUT_ROOT}/${backend}_run${trial_index}"
    mkdir -p "${run_dir}"
    python3 tools/replay_video_to_shm.py \
      --video "${VIDEO_PATH}" \
      --fps "${FPS}" \
      --duration-sec "${PRODUCER_DURATION_SEC}" \
      --start-frame-id "${start_frame_id}" \
      > "${run_dir}/producer.log" 2>&1 &
    producer_pid=$!
    sleep 1
    set +e
    python3 tools/benchmark_latency.py \
      --mode shared_memory \
      --bridge mock \
      --object-backend "${backend}" \
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

printf 'complete\n' > "${OUTPUT_ROOT}/COMPLETE"
