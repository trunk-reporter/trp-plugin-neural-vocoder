#!/usr/bin/env bash
# Package the plugin with its engine and model as an install archive:
#   dist/neural-vocoder-<version>-<target>.tar.gz, holding neural-vocoder-<version>-<target>/
#     neural-vocoder    the plugin
#     p25voc            the engine
#     lib/              libonnxruntime.so.1, libimbe.so
#     model.onnx        the default model
#     licenses/         ONNX Runtime's and KissFFT's
#     README.md, LICENSE
#
#   scripts/package.sh <target> <plugin binary> <engine build dir> <model.onnx>
#
# The engine build dir is engine/build/<arch> (`make -C engine`, with CROSS=aarch64-linux-gnu- for
# a Raspberry Pi from x86-64).
set -euo pipefail

[ $# -eq 4 ] || { sed -n 2,14p "$0" >&2; exit 2; }
target=$1 plugin=$2 engine=$3 model=$4
root="$(cd "$(dirname "$0")/.." && pwd)"
version=$(sed -n 's/^version = "\(.*\)"/\1/p' "$root/Cargo.toml" | head -1)
name="neural-vocoder-$version-$target"
stage="$root/dist/stage"
dir="$stage/$name"
tp="$root/engine/third_party"
ort=$(ls -d "$tp"/onnxruntime* | head -1)

rm -rf "$stage"
mkdir -p "$dir/lib" "$dir/licenses"
install -m755 "$plugin" "$dir/neural-vocoder"
install -m755 "$engine/p25voc" "$dir/p25voc"
install -m644 "$engine/lib/libonnxruntime.so.1" "$engine/lib/libimbe.so" "$dir/lib/"
install -m644 "$model" "$dir/model.onnx"
install -m644 "$ort/LICENSE" "$dir/licenses/onnxruntime-LICENSE"
install -m644 "$ort/ThirdPartyNotices.txt" "$dir/licenses/onnxruntime-ThirdPartyNotices.txt"
install -m644 "$tp/kissfft/COPYING" "$dir/licenses/kissfft-COPYING"
install -m644 "$root/README.md" "$root/LICENSE" "$dir/"
mkdir -p "$root/dist"
tar -C "$stage" -czf "$root/dist/$name.tar.gz" "$name"
rm -rf "$stage"
echo "$root/dist/$name.tar.gz"
