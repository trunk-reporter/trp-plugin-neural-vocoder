# Neural vocoder

A [Trunk Recorder Pro](https://github.com/TrunkRecorder/trunk-recorder-pro) plugin that writes a neural-vocoded copy of
every P25 Phase 1 call beside the recorder's own audio: `<call>.neural.wav` (16-bit, 16 kHz) or `<call>.neural.m4a`. The
recorder's own `.wav`, and everything other plugins upload, are unchanged: plugins only watch.

The vocoder is Vocos-IMBE: a small neural network that resynthesizes speech from the IMBE parameters the radio sent, at
16 kHz instead of the classic vocoder's 8 kHz. On 144 real calls from a P25 Phase 1 system it scores a UTMOSv2 (predicted
listener rating) of 1.90, against 1.42 for the classic IMBE vocoder. It runs in `p25voc`, a small C program (ONNX Runtime,
int8 weights), at about 25 times real time on one Raspberry Pi 5 core.

**Turn on `captureFrames` in Recording settings first.** The plugin vocodes from each call's `.sdr` frames file. Without it
every call is skipped with the reason `no frames file: turn on captureFrames in Recording settings`.

## Installing

Linux only for now (x86-64, and 64-bit ARM such as a Raspberry Pi with a 64-bit OS; glibc 2.35 or newer: Debian 12,
Ubuntu 22.04, Raspberry Pi OS bookworm). On the Plugins page, **Install from GitHub…** with
`https://github.com/trunk-reporter/trp-plugin-neural-vocoder`, or from a terminal:

```bash
trunk-pro plugin install https://github.com/trunk-reporter/trp-plugin-neural-vocoder
```

It's installed off: turn it on in Setup, on the Plugins tab.

## Which calls

| Call | Neural audio |
| --- | --- |
| P25 Phase 1 (IMBE) | yes |
| P25 Phase 2, DMR, NXDN | no (skipped: `no IMBE frames`) |
| Analog, ProVoice | no (skipped) |

Frames lost on air, and frames with more than 10 bit errors, are left out (as in training), so the neural copy can be a
little shorter than the call.

## Settings

| Key | Default | |
| --- | --- | --- |
| `format` | `wav` | `wav` or `m4a` (AAC 32 kbps through `ffmpeg`, which must be installed) |
| `threads` | `1` | CPU cores the engine may use for a call (1 to 8) |
| `model` | empty | An `.onnx` model to use instead of the installed `model.onnx` (an absolute path) |

Per system: `enabled` (default on).

Calls are vocoded one at a time on a worker, at low priority (`nice 10`) so the recorder's own work comes first. Calls still
waiting when recording stops are saved and done at the next start.

The neural files are the plugin's own. The recorder's archive cleanup doesn't know them, so they stay until removed.

## What's in the archive

| File | |
| --- | --- |
| `neural-vocoder` | The plugin |
| `p25voc` | The engine. Its source is in [`engine/`](engine/) |
| `lib/libonnxruntime.so.1` | [ONNX Runtime](https://github.com/microsoft/onnxruntime) 1.22 (MIT), as Microsoft builds it |
| `lib/libimbe.so` | op25's IMBE decoder (Pavel Yazev, GPLv3), used for the parameters only |
| `model.onnx` | The Vocos-IMBE model (win480, int8) |
| `licenses/` | ONNX Runtime's and KissFFT's licenses |

## Building

```bash
make -C engine                                   # engine/build/x86_64: fetches its dependencies first
cargo build --release
scripts/package.sh x86_64-unknown-linux-gnu target/release/neural-vocoder engine/build/x86_64 model.onnx
```

For a Raspberry Pi from an x86-64 machine (`aarch64-linux-gnu-gcc`; on Arch `pacman -S aarch64-linux-gnu-gcc`):

```bash
make -C engine CROSS=aarch64-linux-gnu-
cargo build --release --target aarch64-unknown-linux-gnu    # links with aarch64-linux-gnu-gcc (.cargo/config.toml)
scripts/package.sh aarch64-unknown-linux-gnu target/aarch64-unknown-linux-gnu/release/neural-vocoder engine/build/aarch64 model.onnx
```

Built that way the binaries need the build machine's glibc or newer. Releases are built on Ubuntu 22.04, with the plugin
through `cargo zigbuild` against glibc 2.28: push a tag `v<version>` and `.github/workflows/release.yml` does the rest.
`scripts/model.env` names the model it ships and its SHA-256.

## Developing

```bash
cargo test
NEURAL_VOCODER_ENGINE=engine/build/x86_64/p25voc NEURAL_VOCODER_MODEL=/path/to/model.onnx \
    trunk-pro plugin run ./target/debug/neural-vocoder <calls or folder>
```

## License

GPL-3.0-or-later (the IMBE decoder it uses is GPL).
