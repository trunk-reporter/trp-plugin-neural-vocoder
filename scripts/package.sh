#!/usr/bin/env bash
# Package the plugin with its engine and model as an install archive:
#   dist/neural-vocoder-<version>-<target>.tar.gz, holding neural-vocoder/
#     neural-vocoder    the plugin
#     p25voc            the engine (built with -Wl,-rpath,'$ORIGIN/lib')
#     lib/              libonnxruntime.so.1, libimbe.so
#     model.onnx        the default model
#     README.md, LICENSE
#
#   scripts/package.sh <target> <plugin binary> <p25voc> <libonnxruntime.so.1> <libimbe.so> <model.onnx>
#
# The engine and its libraries come from the p25_vocoder repo: `make -C engine bundle` puts them in
# engine/build/<arch>/ (`CROSS=aarch64-linux-gnu-` for a Raspberry Pi from x86-64). The plugin for arm64:
#   cargo zigbuild --release --target aarch64-unknown-linux-gnu.2.28
set -euo pipefail

[ $# -eq 6 ] || { sed -n 2,14p "$0" >&2; exit 2; }
target=$1 plugin=$2 engine=$3 ort=$4 imbe=$5 model=$6
root="$(cd "$(dirname "$0")/.." && pwd)"
version=$(sed -n 's/^version = "\(.*\)"/\1/p' "$root/Cargo.toml" | head -1)
name="neural-vocoder-$version-$target"
stage="$root/dist/$name"

rm -rf "$stage"
mkdir -p "$stage/neural-vocoder/lib"
install -m755 "$plugin" "$stage/neural-vocoder/neural-vocoder"
install -m755 "$engine" "$stage/neural-vocoder/p25voc"
install -m644 "$ort" "$stage/neural-vocoder/lib/libonnxruntime.so.1"
install -m644 "$imbe" "$stage/neural-vocoder/lib/libimbe.so"
install -m644 "$model" "$stage/neural-vocoder/model.onnx"
install -m644 "$root/README.md" "$root/LICENSE" "$stage/neural-vocoder/"
tar -C "$stage" -czf "$root/dist/$name.tar.gz" neural-vocoder
rm -rf "$stage"
echo "$root/dist/$name.tar.gz"
