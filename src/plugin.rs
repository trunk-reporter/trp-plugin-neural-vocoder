//! The plugin: each concluded P25 Phase 1 call's frames go through the engine, one call at a time on a worker thread, and the
//! result is written beside the call as `<call>.neural.wav` (or `.neural.m4a`).

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use trunk_recorder_plugin::{Attempt, CallQueue, ConcludedCall, Host, Manifest, Plugin, QueueOptions, Setup, State, topic};

use crate::config::{Config, Format, SystemConfig};
use crate::engine::{Engine, Error};
use crate::frames;

const VERSION: &str = env!("CARGO_PKG_VERSION");
pub const NO_FRAMES: &str = "no frames file: turn on captureFrames in Recording settings";

/// What the work on each call needs.
struct Ctx {
    capture_dir: PathBuf,
    /// Short names of the systems turned off.
    disabled: HashSet<String>,
    format: Format,
    /// Where the engine's input (and, for M4A, its WAV) is written.
    work_dir: PathBuf,
    engine: Mutex<Engine>,
}

pub struct NeuralVocoder {
    queue: CallQueue,
}

/// A file installed with the plugin, beside its executable.
fn installed(name: &str) -> PathBuf {
    let exe = std::env::current_exe().unwrap_or_default();
    exe.parent().unwrap_or(Path::new(".")).join(name)
}

/// `$var` when set, else the installed `name`.
fn installed_or(var: &str, name: &str) -> PathBuf {
    std::env::var_os(var).map(PathBuf::from).unwrap_or_else(|| installed(name))
}

impl Plugin for NeuralVocoder {
    type Config = Config;
    type SystemConfig = SystemConfig;

    fn manifest() -> Manifest {
        Manifest { name: "Neural vocoder".into(), subscribe: vec![topic::CALL_CONCLUDED.into()], ..trunk_recorder_plugin::manifest!() }
    }

    fn start(host: Host, setup: Setup<Config, SystemConfig>) -> Result<Self, String> {
        let config = setup.config.clone();
        config.validate()?;
        if config.format == Format::M4a && Command::new("ffmpeg").arg("-version").stdout(Stdio::null()).status().is_err() {
            return Err("Audio format: M4A needs ffmpeg installed (or choose WAV)".into());
        }
        let model = if config.model.is_empty() { installed_or("NEURAL_VOCODER_MODEL", "model.onnx") } else { PathBuf::from(&config.model) };
        let mut engine = Engine::new(installed_or("NEURAL_VOCODER_ENGINE", "p25voc"), model.clone(), config.threads, host.clone());
        engine.start()?;

        let work_dir = setup.data_dir.join("work");
        let _ = std::fs::remove_dir_all(&work_dir);
        std::fs::create_dir_all(&work_dir).map_err(|e| format!("cannot make {}: {e}", work_dir.display()))?;
        let disabled = setup.systems.iter().filter(|s| s.config.as_ref().is_some_and(|c| !c.enabled)).map(|s| s.short_name.clone()).collect();
        let ctx = Arc::new(Ctx { capture_dir: setup.capture_dir.clone(), disabled, format: config.format, work_dir, engine: Mutex::new(engine) });

        let opts = QueueOptions { threads: 1, ..QueueOptions::saved_in(&setup.data_dir) };
        let queue = CallQueue::start(host.clone(), opts, move |call| work(&ctx, call));
        let name = model.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        host.info(format!("neural-vocoder {VERSION}: model {name}, {} thread(s), writing <call>.{}", config.threads, config.format.extension()));
        host.status(State::Ok, format!("Vocoding with {name}"));
        Ok(NeuralVocoder { queue })
    }

    fn call_concluded(&mut self, call: ConcludedCall) {
        self.queue.push(call);
    }

    fn shutdown(&mut self, grace: Duration) {
        self.queue.shutdown(grace);
    }
}

/// Why a call isn't one to vocode, if it isn't.
fn skip_reason(ctx: &Ctx, call: &ConcludedCall) -> Option<&'static str> {
    let r = &call.call;
    if ctx.disabled.contains(&r.short_name) {
        Some("system disabled")
    } else if r.audio_type == "analog" {
        Some("analog")
    } else if r.voice_mode == "provoice" {
        Some("ProVoice")
    } else if r.call_length_ms == 0 && r.call_length == 0.0 {
        Some("no audio")
    } else {
        None
    }
}

/// One try at a call.
fn work(ctx: &Ctx, call: &ConcludedCall) -> Attempt {
    if let Some(why) = skip_reason(ctx, call) {
        return Attempt::Skip(why.into());
    }
    let Some(sdr) = frames::find(&call.files.json, &ctx.capture_dir, &call.path) else {
        return Attempt::Skip(NO_FRAMES.into());
    };
    let frames = match frames::read(&sdr) {
        Ok(f) => f,
        Err(e) => return Attempt::Retry(e),
    };
    let r = &call.call;
    let src = r.src_list.first().map_or(0, |s| s.src.max(0) as u32);
    let (dvcf, n) = frames::dvcf(&frames, r.talkgroup, src, r.start_time_ms.max(0) as u64 * 1000);
    if n == 0 {
        return Attempt::Skip("no IMBE frames (not P25 Phase 1)".into());
    }

    let stem = call.path.replace(['/', '\\'], "_");
    let input = ctx.work_dir.join(format!("{stem}.dvcf"));
    if let Err(e) = std::fs::write(&input, dvcf) {
        return Attempt::Retry(format!("cannot write {}: {e}", input.display()));
    }
    let out = ctx.capture_dir.join(format!("{}.{}", call.path, ctx.format.extension()));
    let result = write(ctx, &input, &out, &stem);
    let _ = std::fs::remove_file(&input);
    match result {
        Ok(()) => Attempt::Done { url: String::new() },
        Err(Error::Call(m)) => Attempt::Fail(m),
        Err(Error::Engine(m)) => Attempt::Retry(m),
    }
}

/// Vocodes `input` to `out`, through a temporary file renamed into place so a reader never sees half of one.
fn write(ctx: &Ctx, input: &Path, out: &Path, stem: &str) -> Result<(), Error> {
    if let Some(dir) = out.parent() {
        std::fs::create_dir_all(dir).map_err(|e| Error::Engine(format!("cannot make {}: {e}", dir.display())))?;
    }
    let tmp = out.with_extension(format!("{}.tmp", out.extension().unwrap_or_default().to_string_lossy()));
    let wav = if ctx.format == Format::Wav { tmp.clone() } else { ctx.work_dir.join(format!("{stem}.wav")) };
    let vocoded = ctx.engine.lock().unwrap_or_else(|e| e.into_inner()).vocode(input, &wav);
    let encoded = vocoded.and_then(|_| if ctx.format == Format::M4a { m4a(&wav, &tmp) } else { Ok(()) });
    if ctx.format == Format::M4a {
        let _ = std::fs::remove_file(&wav);
    }
    if let Err(e) = encoded {
        let _ = std::fs::remove_file(&tmp);
        return Err(e);
    }
    std::fs::rename(&tmp, out).map_err(|e| Error::Engine(format!("cannot write {}: {e}", out.display())))
}

/// AAC in an M4A, as the recorder's own M4As are (mono, 32 kbps is plenty for 16 kHz speech).
fn m4a(wav: &Path, out: &Path) -> Result<(), Error> {
    let status = Command::new("nice")
        .args(["-n", "10", "ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-i"])
        .arg(wav)
        .args(["-c:a", "aac", "-b:a", "32k", "-f", "mp4"])
        .arg(out)
        .stdout(Stdio::null())
        .status()
        .map_err(|e| Error::Engine(format!("cannot run ffmpeg: {e}")))?;
    if status.success() { Ok(()) } else { Err(Error::Engine(format!("ffmpeg failed ({status})"))) }
}
