#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PROJECT_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/../.." && pwd)
RKNN_INCLUDE_DIR=${RKNN_INCLUDE_DIR:-/home/orangepi/Downloads/xsmart_upper_native/native/include}
RGA_INCLUDE_DIR=${RGA_INCLUDE_DIR:-/usr/include/rga}
CXX=${CXX:-g++}

if [ ! -f "$RKNN_INCLUDE_DIR/rknn_api.h" ]; then
  echo "rknn_api.h not found under RKNN_INCLUDE_DIR=$RKNN_INCLUDE_DIR" >&2
  exit 2
fi
if [ ! -f "$RGA_INCLUDE_DIR/im2d.h" ]; then
  echo "im2d.h not found under RGA_INCLUDE_DIR=$RGA_INCLUDE_DIR" >&2
  exit 2
fi

mkdir -p "$SCRIPT_DIR/build"
"$CXX" \
  -std=gnu++17 -O3 -Wall -Wextra \
  -DXSMART_HAVE_RGA=1 \
  -I"$RKNN_INCLUDE_DIR" \
  -I"$RGA_INCLUDE_DIR" \
  -I"$SCRIPT_DIR/include" \
  "$SCRIPT_DIR/src/main.cpp" \
  -o "$SCRIPT_DIR/build/lane_rknn_backend" \
  -lrknnrt -lrga -lpthread -lrt

echo "Built $SCRIPT_DIR/build/lane_rknn_backend"
