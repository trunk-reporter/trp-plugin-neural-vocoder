//! The plugin's settings: the form is drawn from these structs' schema.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// The neural audio's file format.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum Format {
    /// WAV (16-bit, 16 kHz)
    #[default]
    Wav,
    /// M4A (AAC, through ffmpeg)
    M4a,
}

impl Format {
    /// The extension of the files written, after the call's base name.
    pub fn extension(self) -> &'static str {
        match self {
            Format::Wav => "neural.wav",
            Format::M4a => "neural.m4a",
        }
    }
}

/// The plugin's settings.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase", default)]
pub struct Config {
    /// Audio format
    ///
    /// Each P25 Phase 1 call gets <call>.neural.wav or <call>.neural.m4a beside its own audio. M4A needs ffmpeg.
    pub format: Format,
    /// Threads
    ///
    /// CPU cores the engine may use for a call. One keeps up with a busy system on a Raspberry Pi 5.
    #[schemars(range(min = 1, max = 8))]
    pub threads: u32,
    /// Model
    ///
    /// An .onnx model file to use instead of the one installed with the plugin. Empty: the installed one.
    pub model: String,
}

impl Default for Config {
    fn default() -> Self {
        Config { format: Format::Wav, threads: 1, model: String::new() }
    }
}

impl Config {
    /// Refuses settings the plugin cannot run with, by the form's own label.
    pub fn validate(&self) -> Result<(), String> {
        if !(1..=8).contains(&self.threads) {
            return Err("Threads: 1 to 8".into());
        }
        if !self.model.is_empty() && !std::path::Path::new(&self.model).is_absolute() {
            return Err("Model: must be an absolute path".into());
        }
        Ok(())
    }
}

/// Settings for each system.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase", default)]
pub struct SystemConfig {
    /// Vocode this system
    ///
    /// Turn off to leave this system's calls out.
    pub enabled: bool,
}

impl Default for SystemConfig {
    fn default() -> Self {
        SystemConfig { enabled: true }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_and_limits() {
        let mut c = Config::default();
        assert!(c.validate().is_ok());
        assert_eq!(c.format.extension(), "neural.wav");
        c.threads = 0;
        assert_eq!(c.validate().unwrap_err(), "Threads: 1 to 8");
        c.threads = 2;
        c.model = "model.onnx".into();
        assert_eq!(c.validate().unwrap_err(), "Model: must be an absolute path");
    }

    #[test]
    fn the_schema_names_the_fields_in_camel_case() {
        let s = trunk_recorder_plugin::schema::schema_for::<Config>();
        let keys: Vec<&String> = s["properties"].as_object().unwrap().keys().collect();
        assert_eq!(keys, ["format", "threads", "model"]);
        let c: Config = serde_json::from_value(serde_json::json!({"format": "m4a"})).unwrap();
        assert_eq!((c.format, c.threads), (Format::M4a, 1));
    }
}
