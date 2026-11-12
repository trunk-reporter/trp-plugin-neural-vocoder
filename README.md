# Neural vocoder

A Trunk Recorder Pro plugin that writes a neural-vocoded copy of every P25 Phase 1 call beside the recorder's own audio:
`<call>.neural.wav` (16-bit, 16 kHz) or `<call>.neural.m4a`. The recorder's own `.wav`, and everything the other plugins
upload, are unchanged: plugins only watch.

The vocoder is the Vocos-IMBE model from the `p25_vocoder` project. It resynthesizes speech from the IMBE parameters the
radio sent, at 16 kHz instead of the classic vocoder's 8 kHz. On real MARCS calls it scores UTMOSv2 1.90 against the classic
IMBE vocoder's 1.42. It runs in `p25voc`, a small C program (ONNX Runtime, int8 weights), at about 25 times real time on one
Raspberry Pi 5 core.

**Turn on `captureFrames` in Recording settings first.** The plugin vocodes from each call's `.sdr` frames file. Without it
every call is skipped with the reason `no frames file: turn on captureFrames in Recording settings`.

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

## Files

The archive's `neural-vocoder/` folder holds the plugin, the engine `p25voc`, its libraries in `lib/` (ONNX Runtime and
`libimbe`, the op25 IMBE decoder used for the parameters only) and `model.onnx`. To build one for a Raspberry Pi from an
x86-64 machine:

```bash
make -C ../legacy/p25_vocoder/engine bundle CROSS=aarch64-linux-gnu-    # needs aarch64-linux-gnu-gcc
cargo build --release --target aarch64-unknown-linux-gnu                # links with aarch64-linux-gnu-gcc (.cargo/config.toml)
E=../legacy/p25_vocoder/engine/build/aarch64
scripts/package.sh aarch64-unknown-linux-gnu target/aarch64-unknown-linux-gnu/release/neural-vocoder \
    $E/p25voc $E/lib/libonnxruntime.so.1 $E/lib/libimbe.so model.onnx
```

That plugin needs glibc 2.39 (Debian 13 / Raspberry Pi OS trixie). For older systems build it with
`cargo zigbuild --release --target aarch64-unknown-linux-gnu.2.28` (cargo-zigbuild and zig) instead; the engine needs 2.38.

The neural files are the plugin's own. The recorder's archive cleanup doesn't know them, so they stay until removed.

## Developing

```bash
cargo test
NEURAL_VOCODER_ENGINE=/path/to/p25voc NEURAL_VOCODER_MODEL=/path/to/model.onnx \
    trunk-pro plugin run ./target/debug/neural-vocoder <calls or folder>
```

## License

GPL-3.0-or-later (the IMBE decoder it uses is GPL).
