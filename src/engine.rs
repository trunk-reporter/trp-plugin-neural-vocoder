//! The engine: `p25voc -l`, a C program that loads the model once and vocodes the calls it's given, one line each on stdin.
//! It runs as a child process at low priority, started on first use and again after it dies.

use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

use trunk_recorder_plugin::Host;

/// Why a call wasn't vocoded.
#[derive(Debug)]
pub enum Error {
    /// The engine couldn't vocode this call (its reason is in the log).
    Call(String),
    /// The engine itself failed: worth trying again.
    Engine(String),
}

pub struct Engine {
    exe: PathBuf,
    model: PathBuf,
    threads: u32,
    host: Host,
    proc: Option<Proc>,
}

struct Proc {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

impl Proc {
    /// Sends one line; the answer.
    fn ask(&mut self, line: &str) -> Option<String> {
        let mut answer = String::new();
        self.stdin.write_all(line.as_bytes()).and_then(|_| self.stdin.flush()).ok()?;
        match self.stdout.read_line(&mut answer) {
            Ok(n) if n > 0 => Some(answer.trim().to_string()),
            _ => None,
        }
    }
}

impl Drop for Proc {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl Engine {
    pub fn new(exe: PathBuf, model: PathBuf, threads: u32, host: Host) -> Engine {
        Engine { exe, model, threads, host, proc: None }
    }

    /// Starts the engine and waits for its model to load: why it couldn't, if it couldn't.
    pub fn start(&mut self) -> Result<(), String> {
        if !self.exe.is_file() {
            return Err(format!("the engine {} isn't there", self.exe.display()));
        }
        if !self.model.is_file() {
            return Err(format!("the model {} isn't there", self.model.display()));
        }
        let mut proc = self.spawn().map_err(|e| format!("cannot start the engine {}: {e}", self.exe.display()))?;
        match proc.ask("\n").as_deref() {
            Some("ready") => {
                self.proc = Some(proc);
                Ok(())
            }
            _ => Err(format!("the engine stopped loading {} (see the log)", self.model.display())),
        }
    }

    /// Low priority, so the recorder's own work always comes first.
    fn spawn(&self) -> std::io::Result<Proc> {
        let args = [OsArg::from("-m"), self.model.as_os_str().into(), "-t".into(), self.threads.to_string().into(), "-l".into()];
        let mut cmd = Command::new("nice");
        cmd.arg("-n").arg("10").arg(&self.exe).args(&args);
        let child = match pipe(&mut cmd).spawn() {
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => pipe(Command::new(&self.exe).args(&args)).spawn()?,
            r => r?,
        };
        Ok(self.wire(child))
    }

    fn wire(&self, mut child: Child) -> Proc {
        let (stdin, stdout, stderr) = (child.stdin.take().unwrap(), child.stdout.take().unwrap(), child.stderr.take().unwrap());
        let host = self.host.clone();
        // The engine's complaints are warnings; the rest (the IMBE library's banner) is detail.
        std::thread::spawn(move || {
            for line in BufReader::new(stderr).lines().map_while(Result::ok) {
                if line.starts_with("p25voc:") { host.warn(line) } else { host.debug(line) }
            }
        });
        Proc { child, stdin, stdout: BufReader::new(stdout) }
    }

    /// Vocodes the `.dvcf` at `input` to a 16 kHz WAV at `output`; the seconds of audio made.
    pub fn vocode(&mut self, input: &Path, output: &Path) -> Result<f64, Error> {
        let line = format!("{}\t{}\n", input.display(), output.display());
        if line.matches(['\t', '\n']).count() != 2 {
            return Err(Error::Call("a tab or newline in its file name".into()));
        }
        if self.proc.is_none() {
            self.start().map_err(Error::Engine)?;
        }
        let answer = self.proc.as_mut().and_then(|p| p.ask(&line));
        match answer.as_deref() {
            Some(a) if a.starts_with("ok ") => Ok(a[3..].parse().unwrap_or(0.0)),
            Some(_) => Err(Error::Call("the engine couldn't vocode it".into())),
            None => {
                self.proc = None;
                Err(Error::Engine("the engine stopped".into()))
            }
        }
    }
}

type OsArg = std::ffi::OsString;

fn pipe(cmd: &mut Command) -> &mut Command {
    cmd.stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped())
}
