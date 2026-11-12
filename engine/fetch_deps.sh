#!/usr/bin/env bash
# Fetch the engine's third-party sources and libraries into engine/third_party/ (not in git):
#   onnxruntime[-<arch>]  Microsoft's prebuilt ONNX Runtime C library (MIT) for $ARCH
#   kissfft               KissFFT (BSD-3-Clause), the mixed-radix real FFT
#   imbe_vocoder          op25's IMBE vocoder (Pavel Yazev, GPLv3), from trunk-recorder, with a
#                         public decode_params() patched in (defined in imbe_capi.cpp)
# ARCH defaults to this machine's; another one's ONNX Runtime goes in onnxruntime-<arch> (a cross
# build). Safe to re-run: what is there is left alone.
set -euo pipefail

ORT_VERSION=1.22.0
KISSFFT_TAG=131.1.0
TR_REPO=https://github.com/TrunkRecorder/trunk-recorder
TR_COMMIT=29687b52376258f27a79d7fb05fa8e498967e765
IMBE_SUBDIR=lib/op25_repeater/lib/imbe_vocoder

tp="$(cd "$(dirname "$0")" && pwd)/third_party"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
mkdir -p "$tp"
ARCH=${ARCH:-$(uname -m)}
ort="$tp/onnxruntime"
[ "$ARCH" = "$(uname -m)" ] || ort="$tp/onnxruntime-$ARCH"

if [ ! -f "$ort/include/onnxruntime_c_api.h" ]; then
    case "$ARCH" in
        x86_64) arch=x64 ;;
        aarch64|arm64) arch=aarch64 ;;
        *) echo "no prebuilt ONNX Runtime for $ARCH" >&2; exit 1 ;;
    esac
    name="onnxruntime-linux-$arch-$ORT_VERSION"
    curl -fsSL -o "$tmp/ort.tgz" "https://github.com/microsoft/onnxruntime/releases/download/v$ORT_VERSION/$name.tgz"
    tar -xzf "$tmp/ort.tgz" -C "$tmp"
    mv "$tmp/$name" "$ort"
    echo "fetched $name"
fi

if [ ! -f "$tp/kissfft/kiss_fftr.c" ]; then
    git -c advice.detachedHead=false clone -q --depth 1 --branch "$KISSFFT_TAG" \
        https://github.com/mborgerding/kissfft "$tmp/kissfft"
    mkdir -p "$tp/kissfft"
    cp "$tmp/kissfft"/{kiss_fft.c,kiss_fft.h,kiss_fftr.c,kiss_fftr.h,_kiss_fft_guts.h,kiss_fft_log.h,COPYING} "$tp/kissfft/"
    echo "fetched kissfft $KISSFFT_TAG"
fi

imbe="$tp/imbe_vocoder"
if [ ! -f "$imbe/imbe_vocoder.h" ]; then
    git -C "$tmp" init -q imbe
    git -C "$tmp/imbe" remote add origin "$TR_REPO"
    git -C "$tmp/imbe" sparse-checkout set "$IMBE_SUBDIR"
    git -C "$tmp/imbe" fetch -q --depth 1 --filter=blob:none origin "$TR_COMMIT"
    git -C "$tmp/imbe" checkout -q FETCH_HEAD
    cp -r "$tmp/imbe/$IMBE_SUBDIR" "$imbe"
    echo "fetched $IMBE_SUBDIR @ ${TR_COMMIT:0:7}"
fi
# decode_params(): imbe_decode() without the 8 kHz synthesis.
if ! grep -q decode_params "$imbe/imbe_vocoder.h"; then
    sed -i 's|^\(\s*\)const IMBE_PARAM\* param(void).*$|&\n\1void decode_params(int16_t *frame_vector);|' "$imbe/imbe_vocoder.h"
    grep -q decode_params "$imbe/imbe_vocoder.h" || { echo "failed to patch imbe_vocoder.h" >&2; exit 1; }
fi
