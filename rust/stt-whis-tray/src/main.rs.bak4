use arboard::Clipboard;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{SampleFormat, Stream, StreamConfig};
use device_query::{DeviceQuery, DeviceState, Keycode};
use enigo::{Enigo, KeyboardControllable, Key};
use log::{error, info, warn};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::env;
use std::fs;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};
use thiserror::Error;
use whisper_rs::{
    FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters,
};

const DEFAULT_MODEL_PATH: &str = "J:/whistxt/models/ggml-small-q5_1.bin";
const DEFAULT_HOTKEY: &str = "Ctrl+Shift";
const DEFAULT_POLL_HZ: u64 = 30;
const CUDA_BUILD_ENABLED: bool = cfg!(feature = "cuda");
const MAX_SAMPLES: usize = 16_000 * 120; // ~2 minutes of audio

#[derive(Debug, Error)]
enum AppError {
    #[error("IO: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON: {0}")]
    Json(#[from] serde_json::Error),
    #[error("{0}")]
    Msg(String),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum InjectMode {
    Clipboard,
    Keystroke,
}

impl Default for InjectMode {
    fn default() -> Self {
        InjectMode::Clipboard
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct HotkeyConfig {
    /// Simple hold-to-talk combo. Currently only Ctrl+Shift is supported.
    combo: String,
}

impl Default for HotkeyConfig {
    fn default() -> Self {
        Self {
            combo: DEFAULT_HOTKEY.to_string(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Config {
    #[serde(default = "default_model_path")]
    model_path: String,
    #[serde(default)]
    language: Option<String>,
    #[serde(default)]
    hotkey: HotkeyConfig,
    #[serde(default)]
    inject_mode: InjectMode,
    #[serde(default)]
    append_newline: bool,
    #[serde(default = "default_poll_hz")]
    poll_hz: u64,
    #[serde(default = "default_use_cuda")]
    use_cuda: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            model_path: DEFAULT_MODEL_PATH.to_string(),
            language: None,
            hotkey: HotkeyConfig::default(),
            inject_mode: InjectMode::Clipboard,
            append_newline: false,
            poll_hz: DEFAULT_POLL_HZ,
            use_cuda: true,
        }
    }
}

fn default_model_path() -> String {
    DEFAULT_MODEL_PATH.to_string()
}

fn default_poll_hz() -> u64 {
    DEFAULT_POLL_HZ
}

fn default_use_cuda() -> bool {
    true
}

#[derive(Debug)]
enum SpeechEvent {
    RecordingStarted,
    RecordingStopped,
    Processing,
    Info(String),
    Transcript(String),
    Error(String),
}

#[derive(Debug)]
enum SpeechCommand {
    Start {
        model_path: String,
        language: Option<String>,
    },
    Stop,
    Cancel,
}

fn main() -> Result<(), AppError> {
    init_logging();
    let config_path = default_config_path();
    let config = load_or_init_config(&config_path)?;
    info!(
        "Loaded config from {} (use_cuda: {}, cuda build: {})",
        config_path.display(),
        config.use_cuda,
        CUDA_BUILD_ENABLED
    );
    if config.use_cuda && !CUDA_BUILD_ENABLED {
        warn!("Config requests CUDA but binary not built with --features cuda; recompile with --features cuda for GPU.");
    }

    let (speech_tx, speech_rx) = spawn_speech_runtime().map_err(AppError::Msg)?;

    let overlay_handle = start_overlay();

    start_keyboard_loop(config.clone(), speech_tx.clone());

    // Main loop: handle speech events and inject transcripts.
    app_loop(speech_rx, config, overlay_handle)?;
    Ok(())
}

fn init_logging() {
    let mut builder = env_logger::Builder::from_default_env();
    builder
        .format_timestamp_millis()
        .filter_level(log::LevelFilter::Info)
        .init();
}

fn default_config_path() -> PathBuf {
    if let Ok(appdata) = env::var("APPDATA") {
        PathBuf::from(appdata).join("WhisTray").join("config.json")
    } else {
        PathBuf::from("config.json")
    }
}

fn load_or_init_config(path: &Path) -> Result<Config, AppError> {
    if path.exists() {
        let data = fs::read_to_string(path)?;
        let mut cfg: Config = serde_json::from_str(&data)?;
        // If the stored model path is missing, reset to default local model copy.
        if !Path::new(&cfg.model_path).exists() {
            cfg.model_path = default_model_path();
            let updated = serde_json::to_string_pretty(&cfg)?;
            fs::write(path, updated)?;
        }
        return Ok(cfg);
    }
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let cfg = Config::default();
    let data = serde_json::to_string_pretty(&cfg)?;
    fs::write(path, data)?;
    Ok(cfg)
}

fn start_keyboard_loop(config: Config, speech_tx: Sender<SpeechCommand>) {
    thread::spawn(move || {
        let device_state = DeviceState::new();
        let mut hotkey_down = false;
        let poll = Duration::from_millis(1000 / config.poll_hz.max(1));
        loop {
            let keys = device_state.get_keys();
            let ctrl = keys.contains(&Keycode::LControl)
                || keys.contains(&Keycode::RControl);
            let shift =
                keys.contains(&Keycode::LShift) || keys.contains(&Keycode::RShift);
            let down = ctrl && shift; // current combo support
            if down && !hotkey_down {
                hotkey_down = true;
                let _ = speech_tx.send(SpeechCommand::Start {
                    model_path: config.model_path.clone(),
                    language: config.language.clone(),
                });
                info!("Hotkey pressed: start recording");
            } else if !down && hotkey_down {
                hotkey_down = false;
                let _ = speech_tx.send(SpeechCommand::Stop);
                info!("Hotkey released: stop recording");
            }
            thread::sleep(poll);
        }
    });
}

fn app_loop(
    event_rx: Receiver<SpeechEvent>,
    config: Config,
    overlay: OverlayHandle,
) -> Result<(), AppError> {
    // Block on Ctrl+C; process speech events and inject.
    ctrlc::set_handler(move || {
        info!("Ctrl+C received, exiting.");
        std::process::exit(0);
    })
    .map_err(|e| AppError::Msg(format!("failed to set Ctrl+C handler: {e}")))?;

    loop {
        let ev = match event_rx.recv_timeout(Duration::from_millis(50)) {
            Ok(ev) => ev,
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => continue,
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
        };
        match ev {
            SpeechEvent::RecordingStarted => {
                info!("(speech) recording started");
                overlay.send(OverlayMsg::RecordingStart);
            }
            SpeechEvent::RecordingStopped => {
                info!("(speech) recording stopped");
                overlay.send(OverlayMsg::Hide);
            }
            SpeechEvent::Processing => {
                info!("(speech) processing");
                overlay.send(OverlayMsg::Processing);
            }
            SpeechEvent::Info(msg) => info!("(speech) {}", msg),
            SpeechEvent::Error(msg) => error!("(speech) {}", msg),
            SpeechEvent::Transcript(text) => {
                info!("(speech) transcript len={}", text.len());
                let to_inject = if config.append_newline {
                    format!("{text}\n")
                } else {
                    text
                };
                overlay.send(OverlayMsg::Transcript(to_inject.clone()));
                if let Err(e) = inject_text(&to_inject, config.inject_mode.clone()) {
                    error!("inject failed: {}", e);
                }
            }
        }
    }
    Ok(())
}

fn inject_text(text: &str, mode: InjectMode) -> Result<(), String> {
    if text.trim().is_empty() {
        return Ok(());
    }
    // Small delay to allow focus to return after hotkey release.
    thread::sleep(Duration::from_millis(30));
    match mode {
        InjectMode::Clipboard => inject_via_clipboard(text),
        InjectMode::Keystroke => inject_via_keystrokes(text),
    }
}

fn inject_via_clipboard(text: &str) -> Result<(), String> {
    let mut clipboard = Clipboard::new().map_err(|e| format!("clipboard: {e}"))?;
    let prior = clipboard.get_text().ok();
    clipboard
        .set_text(text.to_string())
        .map_err(|e| format!("set clipboard: {e}"))?;
    // Paste via Ctrl+V
    let mut enigo = Enigo::new();
    enigo.key_down(Key::Control);
    enigo.key_click(Key::Layout('v'));
    enigo.key_up(Key::Control);
    // Best-effort restore
    if let Some(old) = prior {
        let _ = clipboard.set_text(old);
    }
    Ok(())
}

fn inject_via_keystrokes(text: &str) -> Result<(), String> {
    let mut enigo = Enigo::new();
    enigo.key_sequence(text);
    Ok(())
}

fn spawn_speech_runtime(
) -> Result<(Sender<SpeechCommand>, Receiver<SpeechEvent>), String> {
    let (cmd_tx, cmd_rx) = mpsc::channel::<SpeechCommand>();
    let (evt_tx, evt_rx) = mpsc::channel::<SpeechEvent>();
    thread::spawn(move || {
        let mut stream: Option<Stream> = None;
        let mut last_sr: u32 = 16_000;
        let mut last_channels: u16 = 1;
        let mut ctx: Option<WhisperContext> = None;
        let mut ctx_model: Option<String> = None;
        let mut lang_opt: Option<String> = None;
        let audio_buf = Arc::new(Mutex::new(Vec::<f32>::new()));
        let host = cpal::default_host();

        let load_ctx = |path: &str,
                        cached_path: &mut Option<String>,
                        cached_ctx: &mut Option<WhisperContext>| {
            if cached_path.as_deref() == Some(path) && cached_ctx.is_some() {
                return Ok(());
            }
            let params = WhisperContextParameters::default();
            match WhisperContext::new_with_params(path, params) {
                Ok(c) => {
                    *cached_ctx = Some(c);
                    *cached_path = Some(path.to_string());
                    Ok(())
                }
                Err(e) => Err(format!("failed to load model: {e}")),
            }
        };

        let transcribe = |ctx: &WhisperContext,
                          audio: &[f32],
                          sr: u32,
                          language: Option<String>|
         -> Result<String, String> {
            let mut state = ctx.create_state().map_err(|e| format!("state: {e}"))?;
            let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
            let threads = std::thread::available_parallelism()
                .unwrap_or_else(|_| NonZeroUsize::new(2).unwrap())
                .get();
            params.set_n_threads(threads as i32);
            if let Some(lang) = language.as_deref() {
                params.set_language(Some(lang));
            } else {
                params.set_language(None);
            }
            params.set_translate(false);
            params.set_print_realtime(false);
            params.set_print_progress(false);
            params.set_print_timestamps(false);

            let pcm = if sr == 16_000 {
                audio.to_vec()
            } else {
                resample_to_16k(audio, sr)
            };
            state
                .full(params, &pcm)
                .map_err(|e| format!("transcribe: {e}"))?;
            let num_segments = state
                .full_n_segments()
                .map_err(|e| format!("segments: {e}"))?;
            let mut out = String::new();
            for i in 0..num_segments {
                out.push_str(&state.full_get_segment_text(i).unwrap_or_default());
            }
            Ok(out.trim().to_string())
        };

        while let Ok(cmd) = cmd_rx.recv() {
            match cmd {
                SpeechCommand::Start { model_path, language } => {
                    lang_opt = language;
                    if !Path::new(&model_path).exists() {
                        let _ = evt_tx.send(SpeechEvent::Error(format!(
                            "Model not found at {model_path}"
                        )));
                        continue;
                    }
                    if let Some(mut buf) = audio_buf.try_lock() {
                        buf.clear();
                    }
                    if stream.is_some() {
                        let _ = evt_tx.send(SpeechEvent::RecordingStarted);
                        continue;
                    }
                    let device = match host.default_input_device() {
                        Some(d) => d,
                        None => {
                            let _ = evt_tx.send(SpeechEvent::Error(
                                "No input device available".into(),
                            ));
                            continue;
                        }
                    };
                    let supported = match device.default_input_config() {
                        Ok(c) => c,
                        Err(e) => {
                            let _ = evt_tx.send(SpeechEvent::Error(format!(
                                "Failed to get input config: {e}"
                            )));
                            continue;
                        }
                    };
                    last_sr = supported.sample_rate().0;
                    last_channels = supported.channels();
                    let stream_config: StreamConfig = supported.clone().into();
                    let err_tx = evt_tx.clone();
                    let buf_clone = audio_buf.clone();
                    let build_stream = |sample_format: SampleFormat| -> Result<Stream, String> {
                        match sample_format {
                            SampleFormat::F32 => device
                                .build_input_stream(
                                    &stream_config,
                                    move |data: &[f32], _| {
                                        push_samples(&buf_clone, data, last_channels);
                                    },
                                    move |e| {
                                        let _ = err_tx.send(SpeechEvent::Error(format!(
                                            "Input stream error: {e}"
                                        )));
                                    },
                                    None,
                                )
                                .map_err(|e| format!("Failed to build input stream: {e}")),
                            SampleFormat::I16 => device
                                .build_input_stream(
                                    &stream_config,
                                    move |data: &[i16], _| {
                                        let mut f32buf: Vec<f32> =
                                            data.iter().map(|s| *s as f32 / i16::MAX as f32).collect();
                                        push_samples(&buf_clone, &f32buf, last_channels);
                                    },
                                    move |e| {
                                        let _ = err_tx.send(SpeechEvent::Error(format!(
                                            "Input stream error: {e}"
                                        )));
                                    },
                                    None,
                                )
                                .map_err(|e| format!("Failed to build input stream: {e}")),
                            SampleFormat::U16 => device
                                .build_input_stream(
                                    &stream_config,
                                    move |data: &[u16], _| {
                                        let mut f32buf: Vec<f32> = data
                                            .iter()
                                            .map(|s| (*s as f32 / u16::MAX as f32) * 2.0 - 1.0)
                                            .collect();
                                        push_samples(&buf_clone, &f32buf, last_channels);
                                    },
                                    move |e| {
                                        let _ = err_tx.send(SpeechEvent::Error(format!(
                                            "Input stream error: {e}"
                                        )));
                                    },
                                    None,
                                )
                                .map_err(|e| format!("Failed to build input stream: {e}")),
                            _ => Err(format!(
                                "Unsupported input sample format: {:?}",
                                sample_format
                            )),
                        }
                    };

                    match build_stream(supported.sample_format()) {
                        Ok(s) => {
                            if let Err(e) = s.play() {
                                let _ = evt_tx.send(SpeechEvent::Error(format!(
                                    "Failed to start capture: {e}"
                                )));
                            } else {
                                stream = Some(s);
                                let _ = evt_tx.send(SpeechEvent::RecordingStarted);
                            }
                        }
                        Err(e) => {
                            let _ = evt_tx.send(SpeechEvent::Error(e));
                        }
                    }
                    if let Err(e) = load_ctx(&model_path, &mut ctx_model, &mut ctx) {
                        let _ = evt_tx.send(SpeechEvent::Error(e));
                    }
                }
                SpeechCommand::Stop | SpeechCommand::Cancel => {
                    if stream.is_some() {
                        stream = None;
                        let _ = evt_tx.send(SpeechEvent::RecordingStopped);
                    }
                    let samples = {
                        if let Some(mut guard) = audio_buf.try_lock() {
                            let data = guard.clone();
                            guard.clear();
                            data
                        } else {
                            Vec::new()
                        }
                    };
                    let _ = evt_tx.send(SpeechEvent::Info(format!(
                        "Captured {} samples @ {} Hz",
                        samples.len(),
                        last_sr
                    )));
                    if samples.is_empty() {
                        continue;
                    }
                    let _ = evt_tx.send(SpeechEvent::Processing);
                    if let Some(ctx_loaded) = ctx.as_ref() {
                        match transcribe(ctx_loaded, &samples, last_sr, lang_opt.clone()) {
                            Ok(t) => {
                                let _ = evt_tx.send(SpeechEvent::Info(format!(
                                    "Transcript length: {} chars",
                                    t.len()
                                )));
                                let _ = evt_tx.send(SpeechEvent::Transcript(t));
                            }
                            Err(e) => {
                                let _ = evt_tx.send(SpeechEvent::Error(e));
                            }
                        }
                    } else {
                        let _ = evt_tx.send(SpeechEvent::Error(
                            "Model not loaded; cannot transcribe".into(),
                        ));
                    }
                }
            }
        }
    });
    Ok((cmd_tx, evt_rx))
}

fn push_samples(buf: &Arc<Mutex<Vec<f32>>>, data: &[f32], channels: u16) {
    let mut guard = match buf.try_lock() {
        Some(g) => g,
        None => return,
    };
    for frame in data.chunks(channels as usize) {
        let mut sum = 0.0f32;
        for s in frame {
            sum += *s;
        }
        guard.push(sum / channels as f32);
        if guard.len() > MAX_SAMPLES {
            let drop = guard.len() - MAX_SAMPLES;
            guard.drain(0..drop);
        }
    }
}

fn resample_to_16k(samples: &[f32], from_rate: u32) -> Vec<f32> {
    if from_rate == 16_000 || samples.is_empty() {
        return samples.to_vec();
    }
    let ratio = 16_000f32 / from_rate as f32;
    let new_len = (samples.len() as f32 * ratio).ceil() as usize;
    let mut out = Vec::with_capacity(new_len);
    for i in 0..new_len {
        let src_pos = i as f32 / ratio;
        let idx = src_pos.floor() as usize;
        let frac = src_pos - idx as f32;
        let s0 = *samples.get(idx).unwrap_or(&0.0);
        let s1 = *samples.get(idx + 1).unwrap_or(&s0);
        out.push(s0 + (s1 - s0) * frac);
    }
    out
}

// ----- Overlay (recording HUD) -----

enum OverlayMsg {
    RecordingStart,
    Processing,
    Transcript(String),
    Hide,
}

struct OverlayHandle;

impl OverlayHandle {
    fn send(&self, _msg: OverlayMsg) {}
}

fn start_overlay() -> OverlayHandle {
    // Overlay temporarily disabled to avoid Win11-only wgpu/WinRT loader deps on Win10.
    OverlayHandle
}
