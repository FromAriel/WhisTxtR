#[cfg(feature = "speech_input")]
use cpal::{
    traits::{DeviceTrait, HostTrait, StreamTrait},
    SampleFormat, Stream, StreamConfig,
};
use eframe::{egui, App};
use egui::text::{LayoutJob, TextFormat};
use egui::{
    Align, Color32, ColorImage, FontData, FontDefinitions, FontFamily, FontId, FontTweak, Layout,
    RichText, TextStyle, TextureHandle, Visuals,
};
use egui_extras::{Size, StripBuilder};
use image::AnimationDecoder;
use rodio::{Decoder, OutputStream, Sink};
use std::collections::{HashMap, HashSet, VecDeque};
use std::env;
use std::fs;
use std::io::{BufReader, Cursor};
#[cfg(feature = "speech_input")]
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, Sender, TryRecvError};
use std::sync::Arc;
#[cfg(feature = "speech_input")]
use std::sync::Mutex;
use std::thread;
use std::time::{Duration, Instant};
use webp_animation::Decoder as WebpAnimDecoder;
#[cfg(feature = "speech_input")]
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};
use zork_core::{
    add_door, add_guarded_exit, add_npc_link, add_npc_node, add_object_to_room, add_room_action,
    add_room_variant, append_builder_change, apply_builder_action, apply_object_action,
    blank_world, build_world_vocab, canonical_map_path, carve_exit, choose_dialog_option,
    clone_object, clone_room, describe_object, describe_room, export_room, file_sha256,
    import_room, init_engine, lint_world_with_assets, load_asset_manifest, load_canonical_map,
    load_from_file as load_save_from_file, load_parser_data, load_world, match_intent,
    merge_seed_into_parser_files, move_object, move_player, normalize_id, normalize_player_input,
    objects_visible_in_room, process_world_events, rename_object, rename_room, resolve_room_assets,
    room_tts_path, save_to_file as save_to_file_core, save_world, seed_actor_runtime,
    set_global_flag, set_room_ambient, set_room_ldesc, set_room_note, set_room_sdesc,
    set_start_room, start_dialog, update_counter, update_object_flags, upsert_npc,
    write_vocab_seed, AssetManifest, BuilderAction, BuilderChange, CounterOp, DescribeOptions,
    DialogView, EngineConfig, EngineState, LintIssue, LintLevel, MapEdgeState, ParseContext,
    ParserData, ParserFiles, ScoredIntent, WorldNpc,
};

#[derive(Clone)]
struct StatusLine {
    message: String,
    success: bool,
}

#[derive(Clone)]
enum LogEntry {
    Plain(String),
    Styled { job: LayoutJob, plain: String },
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum SceneMode {
    Auto,
    ImageOnly,
    Off,
}

#[derive(Default)]
struct ParsedEffects {
    set_flags: Vec<String>,
    clear_flags: Vec<String>,
    counter_changes: Vec<String>,
    give_items: Vec<String>,
    take_items: Vec<String>,
    credits: Option<(i64, bool)>,
}

fn parse_credits_change(token: &str) -> Option<(i64, bool)> {
    let up = token.trim().to_uppercase();
    if !up.starts_with("CREDITS") {
        return None;
    }
    let rest = up.trim_start_matches("CREDITS");
    if rest.is_empty() {
        return None;
    }
    let mut absolute = false;
    let val = if let Some(num) = rest.strip_prefix("+=") {
        num.trim().parse::<i64>().ok()?
    } else if let Some(num) = rest.strip_prefix("-=") {
        -(num.trim().parse::<i64>().ok()?)
    } else if let Some(num) = rest.strip_prefix('+') {
        num.trim().parse::<i64>().ok()?
    } else if let Some(num) = rest.strip_prefix('-') {
        -(num.trim().parse::<i64>().ok()?)
    } else if let Some(num) = rest.strip_prefix('=') {
        absolute = true;
        num.trim().parse::<i64>().ok()?
    } else {
        return None;
    };
    Some((val, absolute))
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum TtsMode {
    FirstTime,
    Always,
    Never,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum PaletteId {
    EvgaGreen,
    AmberCrt,
    ArcticCrt,
    Qb45,
    Mono,
    Lila,
    Swrdfsh,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum SpeechUiState {
    Idle,
    Recording,
    Processing,
}

#[allow(dead_code)]
#[derive(Debug)]
enum SpeechEvent {
    RecordingStarted,
    RecordingStopped,
    Processing,
    Info(String),
    Transcript(String),
    Error(String),
}

#[allow(dead_code)]
#[derive(Debug)]
enum SpeechCommand {
    Start {
        model_path: String,
        language: Option<String>,
    },
    Stop,
    Cancel,
}

#[derive(Clone, Copy)]
struct ThemePalette {
    name: &'static str,
    fg: Color32,
    fg_contrast: Color32,
    panel: Color32,
    widget_inactive: Color32,
    widget_hover: Color32,
    widget_active: Color32,
    success: Color32,
    failure: Color32,
    map_bg: Color32,
    map_edge: Color32,
    map_node_seen: Color32,
    map_node_current: Color32,
}

const PALETTES: [PaletteId; 7] = [
    PaletteId::EvgaGreen,
    PaletteId::AmberCrt,
    PaletteId::ArcticCrt,
    PaletteId::Qb45,
    PaletteId::Mono,
    PaletteId::Lila,
    PaletteId::Swrdfsh,
];

impl PaletteId {
    fn as_str(&self) -> &'static str {
        match self {
            PaletteId::EvgaGreen => "evga_green",
            PaletteId::AmberCrt => "amber_crt",
            PaletteId::ArcticCrt => "arctic_crt",
            PaletteId::Qb45 => "qb45",
            PaletteId::Mono => "mono",
            PaletteId::Lila => "lila",
            PaletteId::Swrdfsh => "swrdfsh",
        }
    }

    fn data(&self) -> ThemePalette {
        match self {
            PaletteId::EvgaGreen => ThemePalette {
                name: "EVGA Green",
                fg: Color32::from_rgb(0x00, 0xFF, 0x80),
                fg_contrast: Color32::from_rgb(0xFF, 0xB6, 0xE6), // light magenta contrast
                panel: Color32::from_rgb(0x05, 0x0A, 0x05),
                widget_inactive: Color32::from_rgb(0x0A, 0x14, 0x0A),
                widget_hover: Color32::from_rgb(0x14, 0x28, 0x14),
                widget_active: Color32::from_rgb(0x1E, 0x3C, 0x1E),
                success: Color32::from_rgb(0x00, 0xC8, 0x78),
                failure: Color32::from_rgb(0xDC, 0x3C, 0x3C),
                map_bg: Color32::from_rgba_unmultiplied(0x00, 0x20, 0x18, 220),
                map_edge: Color32::from_rgb(0x00, 0xC8, 0x8C),
                map_node_seen: Color32::from_rgb(0x00, 0xB4, 0x78),
                map_node_current: Color32::from_rgb(0x00, 0xFF, 0x80),
            },
            PaletteId::AmberCrt => ThemePalette {
                name: "Amber CRT",
                fg: Color32::from_rgb(0xFF, 0xB6, 0x54),
                fg_contrast: Color32::from_rgb(0xB5, 0xE5, 0xFF), // cool blue contrast
                panel: Color32::from_rgb(0x0D, 0x09, 0x04),
                widget_inactive: Color32::from_rgb(0x1A, 0x12, 0x09),
                widget_hover: Color32::from_rgb(0x2A, 0x1C, 0x0D),
                widget_active: Color32::from_rgb(0x3A, 0x26, 0x12),
                success: Color32::from_rgb(0xFF, 0xCE, 0x80),
                failure: Color32::from_rgb(0xE0, 0x52, 0x52),
                map_bg: Color32::from_rgba_unmultiplied(0x1A, 0x10, 0x08, 220),
                map_edge: Color32::from_rgb(0xFF, 0xCA, 0x80),
                map_node_seen: Color32::from_rgb(0xD9, 0x99, 0x54),
                map_node_current: Color32::from_rgb(0xFF, 0xB6, 0x54),
            },
            PaletteId::ArcticCrt => ThemePalette {
                name: "Arctic CRT",
                fg: Color32::from_rgb(0x9E, 0xE6, 0xFF),
                fg_contrast: Color32::from_rgb(0xFF, 0xCF, 0xA8), // warm peach contrast
                panel: Color32::from_rgb(0x05, 0x00, 0x08),
                widget_inactive: Color32::from_rgb(0x0C, 0x08, 0x13),
                widget_hover: Color32::from_rgb(0x14, 0x0D, 0x1D),
                widget_active: Color32::from_rgb(0x1D, 0x13, 0x28),
                success: Color32::from_rgb(0xB8, 0xEF, 0xFF),
                failure: Color32::from_rgb(0xE0, 0x52, 0x7A),
                map_bg: Color32::from_rgba_unmultiplied(0x0C, 0x07, 0x12, 220),
                map_edge: Color32::from_rgb(0xB8, 0xEF, 0xFF),
                map_node_seen: Color32::from_rgb(0x7A, 0xC8, 0xE0),
                map_node_current: Color32::from_rgb(0x9E, 0xE6, 0xFF),
            },
            PaletteId::Qb45 => ThemePalette {
                name: "QB45",
                fg: Color32::from_rgb(0x5D, 0xE0, 0xFF),
                fg_contrast: Color32::from_rgb(0xFF, 0xC4, 0x8A), // soft orange contrast
                panel: Color32::from_rgb(0x00, 0x00, 0x64),
                widget_inactive: Color32::from_rgb(0x02, 0x08, 0x6E),
                widget_hover: Color32::from_rgb(0x0A, 0x12, 0x80),
                widget_active: Color32::from_rgb(0x14, 0x20, 0x93),
                success: Color32::from_rgb(0x6E, 0xE8, 0xFF),
                failure: Color32::from_rgb(0xE0, 0x52, 0x6E),
                map_bg: Color32::from_rgba_unmultiplied(0x00, 0x00, 0x50, 220),
                map_edge: Color32::from_rgb(0x6E, 0xE8, 0xFF),
                map_node_seen: Color32::from_rgb(0x40, 0xB6, 0xD6),
                map_node_current: Color32::from_rgb(0x5D, 0xE0, 0xFF),
            },
            PaletteId::Mono => ThemePalette {
                name: "MONO",
                fg: Color32::from_rgb(0xE8, 0xE8, 0xE8),
                fg_contrast: Color32::from_rgb(0xFF, 0xFF, 0xFF),
                panel: Color32::from_rgb(0x0A, 0x0A, 0x0A),
                widget_inactive: Color32::from_rgb(0x15, 0x15, 0x15),
                widget_hover: Color32::from_rgb(0x20, 0x20, 0x20),
                widget_active: Color32::from_rgb(0x2A, 0x2A, 0x2A),
                success: Color32::from_rgb(0xB0, 0xFF, 0xB0),
                failure: Color32::from_rgb(0xFF, 0x7A, 0x7A),
                map_bg: Color32::from_rgba_unmultiplied(0x12, 0x12, 0x12, 220),
                map_edge: Color32::from_rgb(0xC8, 0xC8, 0xC8),
                map_node_seen: Color32::from_rgb(0x90, 0x90, 0x90),
                map_node_current: Color32::from_rgb(0xE8, 0xE8, 0xE8),
            },
            PaletteId::Lila => ThemePalette {
                name: "Lila",
                fg: Color32::from_rgb(0xE6, 0xD8, 0xFF),
                fg_contrast: Color32::from_rgb(0xCF, 0xFF, 0xE6), // minty contrast
                panel: Color32::from_rgb(0x1A, 0x0E, 0x33),
                widget_inactive: Color32::from_rgb(0x23, 0x14, 0x44),
                widget_hover: Color32::from_rgb(0x2E, 0x1C, 0x55),
                widget_active: Color32::from_rgb(0x39, 0x25, 0x66),
                success: Color32::from_rgb(0xCF, 0xC6, 0xFF),
                failure: Color32::from_rgb(0xE0, 0x5A, 0xA0),
                map_bg: Color32::from_rgba_unmultiplied(0x20, 0x16, 0x3D, 220),
                map_edge: Color32::from_rgb(0xCF, 0xC6, 0xFF),
                map_node_seen: Color32::from_rgb(0xA8, 0x96, 0xD9),
                map_node_current: Color32::from_rgb(0xE6, 0xD8, 0xFF),
            },
            PaletteId::Swrdfsh => ThemePalette {
                name: "SwrdFsh",
                fg: Color32::from_rgb(0xC8, 0xFF, 0x78),
                fg_contrast: Color32::from_rgb(0xE4, 0xC8, 0xFF), // pale lavender contrast
                panel: Color32::from_rgb(0x01, 0x05, 0x00),
                widget_inactive: Color32::from_rgb(0x09, 0x13, 0x0A),
                widget_hover: Color32::from_rgb(0x12, 0x20, 0x14),
                widget_active: Color32::from_rgb(0x1B, 0x2D, 0x1E),
                success: Color32::from_rgb(0xDA, 0xFF, 0x99),
                failure: Color32::from_rgb(0xE0, 0x52, 0x52),
                map_bg: Color32::from_rgba_unmultiplied(0x0A, 0x13, 0x08, 220),
                map_edge: Color32::from_rgb(0xAA, 0xF0, 0x84),
                map_node_seen: Color32::from_rgb(0x7A, 0xC0, 0x5C),
                map_node_current: Color32::from_rgb(0xC8, 0xFF, 0x78),
            },
        }
    }
}

fn palette_from_str(s: &str) -> Option<PaletteId> {
    match s.to_ascii_lowercase().as_str() {
        "evga_green" | "evga" | "ega" => Some(PaletteId::EvgaGreen),
        "amber_crt" | "amber" => Some(PaletteId::AmberCrt),
        "arctic_crt" | "arctic" => Some(PaletteId::ArcticCrt),
        "qb45" => Some(PaletteId::Qb45),
        "mono" => Some(PaletteId::Mono),
        "lila" => Some(PaletteId::Lila),
        "swrdfsh" | "swordfish" => Some(PaletteId::Swrdfsh),
        _ => None,
    }
}

#[derive(Clone)]
struct SavedSettings {
    tts_mode: TtsMode,
    tts_volume: f32,
    palette: PaletteId,
    scene_mode: SceneMode,
    scene_drag_sensitivity: f32,
    scene_invert_y: bool,
    scene_fov_y_deg: f32,
    show_hints: bool,
    show_system_messages: bool,
    font_regular: Option<String>,
    font_bold: Option<String>,
    font_size_base: f32,
    font_size_bold: f32,
    save_path: String,
    speech_enabled: bool,
    speech_auto_send: bool,
    speech_model_path: String,
    speech_language: Option<String>,
    speech_hotkey: String,
}

impl SavedSettings {
    fn with_defaults(
        default_regular: Option<String>,
        default_bold: Option<String>,
        save_path: String,
    ) -> Self {
        SavedSettings {
            tts_mode: TtsMode::FirstTime,
            tts_volume: 0.8,
            palette: PaletteId::EvgaGreen,
            scene_mode: SceneMode::Auto,
            scene_drag_sensitivity: 0.25,
            scene_invert_y: false,
            scene_fov_y_deg: DEFAULT_FOV_Y_DEG,
            show_hints: true,
            show_system_messages: false,
            font_regular: default_regular,
            font_bold: default_bold,
            font_size_base: DEFAULT_FONT_SIZE_BASE,
            font_size_bold: DEFAULT_FONT_SIZE_BOLD,
            save_path,
            speech_enabled: false,
            speech_auto_send: false,
            speech_model_path: DEFAULT_SPEECH_MODEL_PATH.to_string(),
            speech_language: None,
            speech_hotkey: DEFAULT_SPEECH_HOTKEY.to_string(),
        }
    }
}

fn scene_mode_from_str(s: &str) -> Option<SceneMode> {
    match s.to_ascii_lowercase().as_str() {
        "auto" => Some(SceneMode::Auto),
        "image_only" | "image-only" | "image" => Some(SceneMode::ImageOnly),
        "off" => Some(SceneMode::Off),
        _ => None,
    }
}

fn game_display(id: Option<&str>) -> String {
    let g = match id {
        Some(s) => s.to_ascii_lowercase(),
        None => return "UNKNOWN".into(),
    };
    if g.contains("zork1") || g.contains("zork i") || g.trim() == "z1" {
        "ZORK I".into()
    } else if g.contains("zork2") || g.contains("zork ii") || g.trim() == "z2" {
        "ZORK II".into()
    } else if g.contains("zork3") || g.contains("zork iii") || g.trim() == "z3" {
        "ZORK III".into()
    } else {
        g.to_ascii_uppercase()
    }
}

fn load_settings(
    path: &Path,
    default_regular: Option<String>,
    default_bold: Option<String>,
    default_save_path: String,
) -> SavedSettings {
    let mut settings =
        SavedSettings::with_defaults(default_regular, default_bold, default_save_path);
    let data = match fs::read_to_string(path) {
        Ok(d) => d,
        Err(_) => return settings,
    };
    let parsed: serde_json::Value = match serde_json::from_str(&data) {
        Ok(v) => v,
        Err(_) => return settings,
    };
    if let Some(m) = parsed.get("tts_mode").and_then(|v| v.as_str()) {
        settings.tts_mode = match m.to_ascii_lowercase().as_str() {
            "always" => TtsMode::Always,
            "never" => TtsMode::Never,
            _ => TtsMode::FirstTime,
        };
    }
    if let Some(v) = parsed.get("tts_volume").and_then(|v| v.as_f64()) {
        settings.tts_volume = v.clamp(0.0, 1.0) as f32;
    }
    if let Some(p) = parsed.get("palette").and_then(|v| v.as_str()) {
        if let Some(id) = palette_from_str(p) {
            settings.palette = id;
        }
    }
    if let Some(sm) = parsed.get("scene_mode").and_then(|v| v.as_str()) {
        if let Some(mode) = scene_mode_from_str(sm) {
            settings.scene_mode = mode;
        }
    }
    if let Some(v) = parsed
        .get("scene_drag_sensitivity")
        .and_then(|v| v.as_f64())
    {
        settings.scene_drag_sensitivity = v.clamp(0.05, 1.0) as f32;
    }
    if let Some(v) = parsed.get("scene_invert_y").and_then(|v| v.as_bool()) {
        settings.scene_invert_y = v;
    }
    if let Some(v) = parsed.get("scene_fov_y_deg").and_then(|v| v.as_f64()) {
        settings.scene_fov_y_deg = v.clamp(30.0, 110.0) as f32;
    }
    if let Some(v) = parsed.get("show_hints").and_then(|v| v.as_bool()) {
        settings.show_hints = v;
    }
    if let Some(v) = parsed.get("show_system_messages").and_then(|v| v.as_bool()) {
        settings.show_system_messages = v;
    }
    if let Some(v) = parsed.get("font_regular").and_then(|v| v.as_str()) {
        settings.font_regular = Some(v.to_string());
    }
    if let Some(v) = parsed.get("font_bold").and_then(|v| v.as_str()) {
        settings.font_bold = Some(v.to_string());
    }
    if let Some(v) = parsed.get("font_size_base").and_then(|v| v.as_f64()) {
        settings.font_size_base = v.clamp(10.0, 32.0) as f32;
    }
    if let Some(v) = parsed.get("font_size_bold").and_then(|v| v.as_f64()) {
        settings.font_size_bold = v.clamp(10.0, 32.0) as f32;
    }
    if let Some(v) = parsed.get("save_path").and_then(|v| v.as_str()) {
        settings.save_path = v.to_string();
    }
    if let Some(v) = parsed.get("speech_enabled").and_then(|v| v.as_bool()) {
        settings.speech_enabled = v;
    }
    if let Some(v) = parsed.get("speech_auto_send").and_then(|v| v.as_bool()) {
        settings.speech_auto_send = v;
    }
    if let Some(v) = parsed.get("speech_model_path").and_then(|v| v.as_str()) {
        settings.speech_model_path = v.to_string();
    }
    if let Some(v) = parsed.get("speech_language").and_then(|v| v.as_str()) {
        settings.speech_language = Some(v.to_string());
    }
    if let Some(v) = parsed.get("speech_hotkey").and_then(|v| v.as_str()) {
        settings.speech_hotkey = v.to_string();
    }
    settings
}

fn save_settings(path: &Path, settings: &SavedSettings) {
    let mode_str = match settings.tts_mode {
        TtsMode::Always => "always",
        TtsMode::Never => "never",
        TtsMode::FirstTime => "first_time",
    };
    let payload = serde_json::json!({
        "tts_mode": mode_str,
        "tts_volume": settings.tts_volume.clamp(0.0, 1.0),
        "palette": settings.palette.as_str(),
        "scene_mode": match settings.scene_mode {
            SceneMode::Auto => "auto",
            SceneMode::ImageOnly => "image_only",
            SceneMode::Off => "off",
        },
        "scene_drag_sensitivity": settings.scene_drag_sensitivity,
        "scene_invert_y": settings.scene_invert_y,
        "scene_fov_y_deg": settings.scene_fov_y_deg,
        "show_hints": settings.show_hints,
        "show_system_messages": settings.show_system_messages,
        "font_regular": settings.font_regular,
        "font_bold": settings.font_bold,
        "font_size_base": settings.font_size_base,
        "font_size_bold": settings.font_size_bold,
        "save_path": settings.save_path,
        "speech_enabled": settings.speech_enabled,
        "speech_auto_send": settings.speech_auto_send,
        "speech_model_path": settings.speech_model_path,
        "speech_language": settings.speech_language,
        "speech_hotkey": settings.speech_hotkey,
    });
    if let Some(parent) = path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    let _ = fs::write(
        path,
        serde_json::to_string_pretty(&payload).unwrap_or_default(),
    );
}

const FONT_FAMILY_BASE: &str = "ZorkBase";
const FONT_FAMILY_BOLD: &str = "ZorkBold";
const FONT_FAMILY_BOLD_UI: &str = "ZorkBoldUi";
const DEFAULT_FONT_SIZE_BASE: f32 = 24.0;
const DEFAULT_FONT_SIZE_BOLD: f32 = 15.0;
const BOLD_Y_OFFSET_PX: f32 = 2.0;
const MAX_PITCH_DEG: f32 = 85.0;
const DEFAULT_FOV_Y_DEG: f32 = 60.0;
const DEFAULT_YAW_DEG: f32 = 180.0;
const DEFAULT_SPEECH_MODEL_PATH: &str = "../out/models/whisper/small.pt";
const DEFAULT_SPEECH_HOTKEY: &str = "LeftCtrl";
const SPEECH_BUILD_ENABLED: bool = cfg!(feature = "speech_input");

#[derive(Clone)]
struct FontEntry {
    name: String,
    path: PathBuf,
}

#[derive(Clone)]
struct AnimatedFrame {
    image: ColorImage,
    duration_secs: f32,
}

#[derive(Clone)]
struct AnimatedFrames {
    name: String,
    frames: Vec<AnimatedFrame>,
}

const PANO_SEAM_BLEND_PX: usize = 10;
const PANO_CACHE_HOT_LIMIT: usize = 5;
const PANO_CACHE_PREFETCH_LIMIT: usize = 5;
const PANO_QUEUE_LIMIT: usize = 10;
const PANO_MEM_BUDGET_BYTES: usize = 5 * 1024 * 1024 * 1024;
const PANO_JOB_TIMEOUT_SECS: u64 = 20;

fn apply_pano_seam_blend(image: &mut egui::ColorImage, blend_px: usize) {
    let w = image.size[0];
    let h = image.size[1];
    if blend_px == 0 || w < blend_px * 2 {
        return;
    }
    let blend = |a: u8, b: u8, alpha: f32, inv: f32| -> u8 {
        ((a as f32) * inv + (b as f32) * alpha + 0.5) as u8
    };
    for x in 0..blend_px {
        // Fade from original edge toward the opposite edge, leaving a bit of original at the boundary.
        let alpha = (x as f32 + 1.0) / (blend_px as f32 + 1.0);
        let inv = 1.0 - alpha;
        for y in 0..h {
            let li = y * w + x;
            let ri = y * w + (w - 1 - x);
            let l = image.pixels[li].to_array();
            let r = image.pixels[ri].to_array();
            let left = egui::Color32::from_rgba_unmultiplied(
                blend(l[0], r[0], alpha, inv),
                blend(l[1], r[1], alpha, inv),
                blend(l[2], r[2], alpha, inv),
                blend(l[3], r[3], alpha, inv),
            );
            let right = egui::Color32::from_rgba_unmultiplied(
                blend(r[0], l[0], alpha, inv),
                blend(r[1], l[1], alpha, inv),
                blend(r[2], l[2], alpha, inv),
                blend(r[3], l[3], alpha, inv),
            );
            image.pixels[li] = left;
            image.pixels[ri] = right;
        }
    }
}

fn apply_pano_seam_blend_frames(frames: &mut [AnimatedFrame]) {
    for frame in frames.iter_mut() {
        apply_pano_seam_blend(&mut frame.image, PANO_SEAM_BLEND_PX);
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[allow(dead_code)]
enum PanoJobPriority {
    Current = 3,
    Destination = 2,
    Neighbor = 1,
    Prefetch = 0,
}

#[derive(Clone)]
struct PanoJob {
    room_id: String,
    name: String,
    source: PanoSource,
    priority: PanoJobPriority,
    generation: u64,
    prefetch: bool,
}

#[derive(Clone)]
struct PanoLoadResult {
    room_id: String,
    name: String,
    frames: Option<Arc<AnimatedFrames>>,
    generation: u64,
    prefetch: bool,
}

#[derive(Clone)]
enum PanoSource {
    WebpBytes(Arc<Vec<u8>>),
    Mp4Path(PathBuf),
}

#[derive(Clone)]
struct PanoCacheEntry {
    name: String,
    first_frame: AnimatedFrame,
    frames: Option<Arc<AnimatedFrames>>,
    size: [usize; 2],
    loading: bool,
    prefetch: bool,
    approx_bytes: usize,
}

fn estimate_frame_bytes(frame: &AnimatedFrame) -> usize {
    frame.image.pixels.len().saturating_mul(4)
}

fn estimate_frames_bytes(frames: &AnimatedFrames) -> usize {
    frames.frames.iter().map(|f| estimate_frame_bytes(f)).sum()
}

#[derive(Clone)]
struct InventoryEntry {
    id: String,
    label: String,
    can_open: bool,
}

impl LogEntry {
    fn to_plain_string(&self) -> String {
        match self {
            LogEntry::Plain(s) => s.clone(),
            LogEntry::Styled { plain, .. } => plain.clone(),
        }
    }
}

struct ZorkEguiApp {
    manifest_status: String,
    manifest: Option<AssetManifest>,
    assets_base: PathBuf,
    parser: Option<ParserData>,
    ctx: ParseContext,
    input: String,
    input_has_focus: bool,
    log: Vec<LogEntry>,
    current_dialog: Option<DialogView>,
    hero_texture: Option<TextureHandle>,
    pano_texture: Option<TextureHandle>,
    pano_size: Option<[usize; 2]>,
    pano_frames: Option<Arc<AnimatedFrames>>,
    pano_frame_index: usize,
    pano_next_switch: f64,
    pano_rel: Option<String>,
    pano_cache: HashMap<String, PanoCacheEntry>,
    pano_cache_hot_order: VecDeque<String>,
    pano_cache_prefetch_order: VecDeque<String>,
    pano_cache_bytes: usize,
    pano_jobs: Vec<PanoJob>,
    pano_inflight: Option<PanoJob>,
    pano_job_started: HashMap<String, Instant>,
    pano_job_attempts: HashMap<String, u8>,
    pano_generation: u64,
    pano_loader_rx: Receiver<PanoLoadResult>,
    pano_loader_tx: Sender<PanoLoadResult>,
    scene_mode: SceneMode,
    scene_yaw_deg: f32,
    scene_yaw_target_deg: f32,
    scene_pitch_deg: f32,
    scene_pitch_target_deg: f32,
    scene_invert_y: bool,
    scene_drag_sensitivity: f32,
    scene_fov_y_deg: f32,
    show_options: bool,
    status: Option<StatusLine>,
    status_time: Option<f64>,
    current_time: f64,
    save_path: String,
    builder_world_path: Option<PathBuf>,
    builder_log_path: Option<PathBuf>,
    show_help_panel: bool,
    rebuild_rx: Option<Receiver<String>>,
    engine: EngineState,
    map_zoom: f32,
    map_pan: egui::Vec2,
    show_inventory: bool,
    inventory_hover: Option<String>,
    show_hints: bool,
    fonts: Vec<FontEntry>,
    font_regular: Option<String>,
    font_bold: Option<String>,
    font_size_base: f32,
    font_size_bold: f32,
    font_error: Option<String>,
    tts_mode: TtsMode,
    tts_volume: f32,
    palette: PaletteId,
    last_move_dir: Option<String>,
    tts_played_rooms: HashSet<String>,
    _tts_stream: Option<OutputStream>,
    tts_handle: Option<rodio::OutputStreamHandle>,
    tts_cancel: Option<Arc<AtomicBool>>,
    tts_sink: Option<Arc<Sink>>,
    speech_enabled: bool,
    speech_auto_send: bool,
    speech_model_path: String,
    speech_language: Option<String>,
    speech_hotkey: String,
    speech_state: SpeechUiState,
    speech_hotkey_down: bool,
    speech_tx: Option<Sender<SpeechCommand>>,
    speech_rx: Option<Receiver<SpeechEvent>>,
    speech_started_at: Option<Instant>,
    settings_path: PathBuf,
    show_system_messages: bool,
    pulse_next: f64,
    pulse_interval: f64,
    pulse_duration: f64,
    pulse_spans: Vec<(usize, usize)>,
    pulse_text: Option<String>,
}

impl App for ZorkEguiApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.apply_palette_visuals(ctx);
        self.current_time = ctx.input(|i| i.time);
        self.poll_rebuild_queue();
        self.poll_speech_queue(ctx);
        self.handle_speech_hotkey(ctx);
        self.handle_hotkeys(ctx);
        self.check_pano_timeouts();
        self.kick_pano_worker();
        self.process_pano_loader(ctx);
        self.smooth_scene_angles(ctx);
        self.refresh_pulse(self.current_time);
        self.schedule_pulse_repaint(ctx, self.current_time);

        if self.show_options {
            let mut open = self.show_options;
            let mut close_now = false;
            let mut saved = false;
            egui::Window::new("Options")
                .collapsible(false)
                .resizable(false)
                .open(&mut open)
                .show(ctx, |ui| {
                    ui.label(RichText::new("Save / Load").underline());
                    ui.add_space(16.0);
                    ui.horizontal(|ui| {
                        if ui.button("Save").clicked() {
                            self.try_save(&self.save_path.clone());
                            saved = true;
                        }
                        if ui.button("Load").clicked() {
                            self.try_load(&self.save_path.clone());
                            saved = true;
                        }
                        ui.separator();
                        ui.label("Path:");
                        ui.text_edit_singleline(&mut self.save_path);
                    });
                    ui.separator();
                    ui.label(RichText::new("Scene / Media").underline());
                    ui.add_space(16.0);
                    egui::ComboBox::from_id_source("scene_mode_combo")
                        .selected_text(match self.scene_mode {
                            SceneMode::Auto => "Auto (prefer pano when present)",
                            SceneMode::ImageOnly => "Images only (hero/thumb)",
                            SceneMode::Off => "Off",
                        })
                        .show_ui(ui, |ui| {
                            ui.selectable_value(
                                &mut self.scene_mode,
                                SceneMode::Auto,
                                "Auto (prefer pano when present)",
                            );
                            ui.selectable_value(
                                &mut self.scene_mode,
                                SceneMode::ImageOnly,
                                "Images only (hero/thumb)",
                            );
                            ui.selectable_value(&mut self.scene_mode, SceneMode::Off, "Off");
                        });
                    ui.horizontal(|ui| {
                        ui.label("Drag:");
                        ui.add(
                            egui::Slider::new(&mut self.scene_drag_sensitivity, 0.05..=1.0)
                                .text("deg/px"),
                        );
                        ui.checkbox(&mut self.scene_invert_y, "Invert Y");
                    });
                    ui.horizontal(|ui| {
                        ui.label("FOV:");
                        ui.add(
                            egui::Slider::new(&mut self.scene_fov_y_deg, 30.0..=110.0).text("deg"),
                        );
                        if ui.button("Reset view").clicked() {
                            self.scene_yaw_deg = 0.0;
                            self.scene_pitch_deg = 0.0;
                            self.scene_yaw_target_deg = self.scene_yaw_deg;
                            self.scene_pitch_target_deg = self.scene_pitch_deg;
                        }
                    });
                    ui.checkbox(&mut self.show_hints, "Show visible items (hints)");
                    ui.horizontal(|ui| {
                        ui.label("Palette:");
                        egui::ComboBox::from_id_source("palette_combo")
                            .selected_text(self.current_palette().name)
                            .show_ui(ui, |ui| {
                                for pal in PALETTES {
                                    if ui
                                        .selectable_value(&mut self.palette, pal, pal.data().name)
                                        .clicked()
                                    {
                                        self.apply_palette_visuals(ctx);
                                    }
                                }
                            });
                    });
                    ui.separator();
                    ui.label(RichText::new("Narrator").underline());
                    ui.add_space(16.0);
                    ui.horizontal(|ui| {
                        ui.label("Narrator:");
                        ui.radio_value(&mut self.tts_mode, TtsMode::FirstTime, "First time");
                        ui.radio_value(&mut self.tts_mode, TtsMode::Always, "Always");
                        ui.radio_value(&mut self.tts_mode, TtsMode::Never, "Never");
                        ui.separator();
                        ui.label("Volume:");
                        ui.add(
                            egui::Slider::new(&mut self.tts_volume, 0.0..=1.0).clamp_to_range(true),
                        );
                    });
                    ui.checkbox(&mut self.show_system_messages, "Show status toasts/logs");
                    ui.separator();
                    ui.label(RichText::new("Speech Input (beta)").underline());
                    ui.add_space(16.0);
                    if !SPEECH_BUILD_ENABLED {
                        ui.colored_label(
                            self.current_palette().failure,
                            "Not available in this build (compile with feature \"speech_input\").",
                        );
                    }
                    let enable_label =
                        format!("Enable speech (hold {})", self.speech_hotkey.trim());
                    ui.add_enabled_ui(SPEECH_BUILD_ENABLED, |ui| {
                        ui.checkbox(&mut self.speech_enabled, enable_label);
                        ui.checkbox(&mut self.speech_auto_send, "Auto-send transcript");
                    });
                    ui.horizontal(|ui| {
                        ui.label("Model path:");
                        ui.text_edit_singleline(&mut self.speech_model_path);
                    });
                    let mut lang_buf = self.speech_language.clone().unwrap_or_default();
                    ui.horizontal(|ui| {
                        ui.label("Language (blank=auto):");
                        ui.text_edit_singleline(&mut lang_buf);
                    });
                    self.speech_language = if lang_buf.is_empty() {
                        None
                    } else {
                        Some(lang_buf)
                    };
                    ui.horizontal(|ui| {
                        ui.label("Hold key:");
                        ui.text_edit_singleline(&mut self.speech_hotkey);
                    });
                    ui.separator();
                    ui.label(RichText::new("Font / Styling").underline());
                    ui.add_space(16.0);
                    ui.horizontal(|ui| {
                        ui.label("Text:");
                        egui::ComboBox::from_id_source("font_regular_combo")
                            .selected_text(self.font_label(self.font_regular.as_deref()))
                            .show_ui(ui, |ui| {
                                ui.selectable_value(
                                    &mut self.font_regular,
                                    None,
                                    "Default (Rusty Knife)",
                                );
                                for f in self.fonts.iter() {
                                    let path_str = f.path.to_string_lossy().to_string();
                                    ui.selectable_value(
                                        &mut self.font_regular,
                                        Some(path_str),
                                        &f.name,
                                    );
                                }
                            });
                        ui.separator();
                        ui.label("Bold:");
                        egui::ComboBox::from_id_source("font_bold_combo")
                            .selected_text(self.font_label(self.font_bold.as_deref()))
                            .show_ui(ui, |ui| {
                                ui.selectable_value(&mut self.font_bold, None, "Match regular");
                                for f in self.fonts.iter() {
                                    let path_str = f.path.to_string_lossy().to_string();
                                    ui.selectable_value(
                                        &mut self.font_bold,
                                        Some(path_str),
                                        &f.name,
                                    );
                                }
                            });
                    });
                    ui.horizontal(|ui| {
                        ui.label("Text size:");
                        ui.add(egui::Slider::new(&mut self.font_size_base, 10.0..=32.0).text("px"));
                        ui.label("Bold size:");
                        ui.add(egui::Slider::new(&mut self.font_size_bold, 10.0..=32.0).text("px"));
                    });
                    if ui.button("Apply font").clicked() {
                        self.apply_fonts(ctx);
                        self.push_plain(format!(
                            "(options) Applied font (text {:.1}px, bold {:.1}px)",
                            self.font_size_base, self.font_size_bold
                        ));
                        saved = true;
                    }
                    if let Some(err) = &self.font_error {
                        ui.colored_label(egui::Color32::RED, err);
                    }
                    if ui.button("Close").clicked() {
                        close_now = true;
                    }
                });
            let should_close = !open || close_now || saved;
            if should_close {
                self.persist_settings();
            }
            self.show_options = !should_close;
        }

        egui::TopBottomPanel::top("header").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.heading("InDEED The GReat Underground Empire");
                ui.separator();
                ui.label(&self.manifest_status);
                if let Some(credits) = self.engine.global_state.counters.get("CREDITS") {
                    ui.separator();
                    ui.label(format!("Credits: {}", credits));
                }
                ui.separator();
                let help_label = if self.show_help_panel {
                    "Hide Help"
                } else {
                    "Help"
                };
                if ui.button(help_label).clicked() {
                    self.show_help_panel = !self.show_help_panel;
                }
            });
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            // Outer vertical split: top 2/3 (scene+context) and bottom 1/3 (transcript+input)
            let exits = self.available_exits();

            StripBuilder::new(ui)
                .size(Size::relative(0.6667)) // top 2/3
                .size(Size::remainder()) // bottom 1/3
                .clip(true)
                .vertical(|mut vstrip| {
                    // ---------- TOP REGION (Scene + Context) ----------
                    vstrip.cell(|ui| {
                        // Horizontal split inside top: left 2/3 scene, right 1/3 context
                        StripBuilder::new(ui)
                            .size(Size::relative(0.6667)) // scene
                            .size(Size::remainder()) // context
                            .horizontal(|mut hstrip| {
                                // Left: Scene
                                hstrip.cell(|ui| {
                                    ui.group(|ui| {
                                        self.render_scene(ui, ctx);
                                    });
                                });

                                // Right: Context
                                hstrip.cell(|ui| {
                                    ui.group(|ui| {
                                        let pretty_room = if let Some(r) = self.ctx.room.as_ref() {
                                            self.prettify_id(r)
                                        } else {
                                            "<none>".into()
                                        };
                                        ui.label(format!("Room: {}", pretty_room));
                                        if let Some(view) = self.current_dialog.clone() {
                                            ui.separator();
                                            ui.label(
                                                RichText::new(format!(
                                                    "Talking to {}",
                                                    self.prettify_id(&view.npc_id)
                                                ))
                                                .strong(),
                                            );
                                            ui.label(view.text.clone());
                                            for (i, opt) in view.options.iter().enumerate() {
                                                let label = format!("[{}] {}", i + 1, opt.prompt);
                                                if ui.button(label).clicked() {
                                                    self.choose_dialog_option_idx(i);
                                                }
                                            }
                                            if view.options.is_empty() {
                                                ui.label("(dialog) [ended]");
                                            }
                                        }
                                        if self.show_inventory {
                                            let available = ui.available_size();
                                            ui.allocate_ui_with_layout(
                                                available,
                                                Layout::top_down(Align::LEFT),
                                                |ui| {
                                                    self.render_inventory_overlay(ui, available, ctx);
                                                },
                                            );
                                        } else {
                                            ui.label("Map");
                                            let available = ui.available_size();
                                            if self.show_help_panel {
                                                egui::ScrollArea::vertical()
                                                    .max_height(available.y)
                                                    .show(ui, |ui| {
                                                        self.render_builder_help(ui);
                                                    });
                                            } else {
                                                let (rect, response) = ui.allocate_exact_size(
                                                    available,
                                                    egui::Sense::drag()
                                                        .union(egui::Sense::hover()),
                                                );
                                                let painter = ui.painter_at(rect);
                                                self.render_map(&painter, rect, &response);
                                                if response.hovered() {
                                                    ui.ctx().set_cursor_icon(egui::CursorIcon::Grab);
                                                }
                                            }
                                        }
                                    });
                                });
                            });
                    });

                    // ---------- BOTTOM REGION (Transcript + Input + whitespace) ----------
                    vstrip.cell(|ui| {
                        // Horizontal split in bottom area:
                        //   - left 2/3: transcript + input
                        //   - right 1/3: direction/action pad
                        StripBuilder::new(ui)
                            .size(Size::relative(0.6667)) // left 2/3
                            .size(Size::remainder()) // right 1/3 pad
                            .horizontal(|mut hstrip| {
                                // Left cell: transcript + input (stacked vertically)
                                hstrip.cell(|ui| {
                                    StripBuilder::new(ui)
                                        .size(Size::remainder()) // transcript
                                        .size(Size::exact(40.0)) // input row
                                        .vertical(|mut bottom_strip| {
                                            // Transcript box
                                            bottom_strip.cell(|ui| {
                                                ui.group(|ui| {
                                                    ui.horizontal(|ui| {
                                                        ui.label("Transcript");
                                                        ui.with_layout(
                                                            Layout::right_to_left(Align::Center),
                                                            |ui| {
                                                                if ui.button("\u{29C9}").clicked() {
                                                                let mut text = String::new();
                                                                for entry in self.log.iter().rev() {
                                                                    text.push_str(&entry.to_plain_string());
                                                                    text.push('\n');
                                                                }
                                                                ui.output_mut(|o| {
                                                                    o.copied_text = text;
                                                                });
                                                                self.push_plain(
                                                                    "(copied transcript to clipboard)",
                                                                );
                                                            }
                                                        },
                                                    );
                                                });
                                                egui::ScrollArea::vertical()
                                                        .auto_shrink([false; 2])
                                                        .stick_to_bottom(true)
                                                        .show(ui, |ui| {
                                                            for entry in self.log.iter().rev() {
                                                                match entry {
                                                                    LogEntry::Plain(s) => {
                                                                        ui.label(s);
                                                                    }
                                                                    LogEntry::Styled { job, .. } => {
                                                                        ui.label(job.clone());
                                                                    }
                                                                }
                                                            }
                                                        });

                                                    // Make transcript group fill its cell
                                                    ui.allocate_space(ui.available_size());
                                                });
                                            });

                                            // Input row at bottom-left
                                            bottom_strip.cell(|ui| {
                                                ui.with_layout(
                                                    Layout::left_to_right(Align::Center),
                                                    |ui| {
                                                        let window_focused =
                                                            ctx.input(|i| i.focused);
                                                        if self.input_has_focus && !window_focused {
                                                            self.input_has_focus = false;
                                                        }
                                                        let resp =
                                                            ui.text_edit_singleline(&mut self.input);
                                                        if self.input_has_focus
                                                            && window_focused
                                                            && !resp.has_focus()
                                                        {
                                                            resp.request_focus();
                                                        }
                                                        if resp.has_focus() {
                                                            self.input_has_focus = true;
                                                        }
                                                        if resp.lost_focus()
                                                            && ui.input(|i| {
                                                                i.key_pressed(egui::Key::Enter)
                                                            })
                                                        {
                                                            self.handle_input(ctx);
                                                        }
                                                        if ui.button("Send").clicked() {
                                                            self.handle_input(ctx);
                                                        }
                                                    },
                                                );
                                            });
                                        });
                                });

                                // Right cell: keypad-style direction/action pad
                                hstrip.cell(|ui| {
                                    ui.group(|ui| {
                                        ui.horizontal(|ui| {
                                            ui.label("Actions / Moves");
                                            ui.with_layout(
                                                Layout::right_to_left(Align::Center),
                                                |ui| {
                                                    if ui.button("\u{23DA}").clicked() {
                                                        self.show_options = true;
                                                    }
                                                },
                                            );
                                        });
                                        ui.add_space(6.0);

                                        let button_size = egui::vec2(64.0, 28.0);
                                        let btn = |ui: &mut egui::Ui,
                                                   label: &str,
                                                   cmd: &str,
                                                   aliases: &[&str],
                                                   exits: &HashSet<String>,
                                                   me: &mut ZorkEguiApp,
                                                   ctx: &egui::Context| {
                                            let enabled = me.has_exit(exits, aliases);
                                            let resp = ui.add_enabled(
                                                enabled,
                                                egui::Button::new(label)
                                                    .min_size(button_size),
                                            );
                                            if resp.clicked() {
                                                me.send_command(cmd, ctx);
                                            }
                                        };

                                        egui::Grid::new("dir_pad_grid")
                                            .spacing(egui::vec2(6.0, 6.0))
                                            .show(ui, |ui| {
                                                // Row 1: HELP IN OUT UP (uses - for up to match numpad key)
                                                let help_resp = ui.add_sized(
                                                    button_size,
                                                    egui::Button::new("HELP"),
                                                );
                                                if help_resp.clicked() {
                                                    self.show_help_panel = !self.show_help_panel;
                                                }
                                                btn(ui, "IN", "in", &["IN", "ENTER"], &exits, self, ctx);
                                                btn(
                                                    ui,
                                                    "OUT",
                                                    "out",
                                                    &["OUT", "EXIT"],
                                                    &exits,
                                                    self,
                                                    ctx,
                                                );
                                                btn(ui, "UP", "up", &["UP", "U", "-"], &exits, self, ctx);
                                                ui.end_row();

                                                // Row 2: NW N NE DOWN (uses + for down to match numpad key)
                                                btn(
                                                    ui,
                                                    "NW",
                                                    "nw",
                                                    &["NW", "NORTHWEST"],
                                                    &exits,
                                                    self,
                                                    ctx,
                                                );
                                                btn(
                                                    ui,
                                                    "N",
                                                    "north",
                                                    &["N", "NORTH"],
                                                    &exits,
                                                    self,
                                                    ctx,
                                                );
                                                btn(
                                                    ui,
                                                    "NE",
                                                    "ne",
                                                    &["NE", "NORTHEAST"],
                                                    &exits,
                                                    self,
                                                    ctx,
                                                );
                                                btn(
                                                    ui,
                                                    "DOWN",
                                                    "down",
                                                    &["DOWN", "D", "+"],
                                                    &exits,
                                                    self,
                                                    ctx,
                                                );
                                                ui.end_row();

                                                // Row 3: W LOOK E INV
                                                btn(ui, "W", "west", &["W", "WEST"], &exits, self, ctx);
                                                // Use custom glyphs for the double-O in LOOK to exercise font icons.
                                                let resp = ui.add(
                                                    egui::Button::new("L\u{25ce}\u{25ce}K")
                                                        .min_size(button_size),
                                                );
                                                if resp.clicked() {
                                                    self.send_command("look", ctx);
                                                }
                                                btn(
                                                    ui,
                                                    "E",
                                                    "east",
                                                    &["E", "EAST"],
                                                    &exits,
                                                    self,
                                                    ctx,
                                                );
                                                let inv_resp = ui.add(
                                                    egui::Button::new("INV")
                                                        .min_size(button_size),
                                                );
                                                if inv_resp.clicked() {
                                                    self.toggle_inventory_overlay();
                                                }
                                                ui.end_row();

                                                // Row 4: SW S SE EXIT
                                                btn(
                                                    ui,
                                                    "SW",
                                                    "southwest",
                                                    &["SW", "SOUTHWEST"],
                                                    &exits,
                                                    self,
                                                    ctx,
                                                );
                                                btn(
                                                    ui,
                                                    "S",
                                                    "south",
                                                    &["S", "SOUTH"],
                                                    &exits,
                                                    self,
                                                    ctx,
                                                );
                                                btn(
                                                    ui,
                                                    "SE",
                                                    "southeast",
                                                    &["SE", "SOUTHEAST"],
                                                    &exits,
                                                    self,
                                                    ctx,
                                                );
                                                let exit_resp = ui.add(
                                                    egui::Button::new("EXIT")
                                                        .min_size(button_size),
                                                );
                                                if exit_resp.clicked() {
                                                    ui.ctx()
                                                        .send_viewport_cmd(egui::ViewportCommand::Close);
                                                }
                                                ui.end_row();
                                            });
                                    });
                                });
                            });
                    });
                });
        });

        // Status toasts rerouted into transcript once, optional inline tag.
        if self.show_system_messages {
            if let (Some(status), Some(t0)) = (&self.status, self.status_time) {
                let age = ctx.input(|i| i.time) - t0;
                if age < 2.5 {
                    let pal = self.current_palette();
                    let color = if status.success {
                        pal.success
                    } else {
                        pal.failure
                    };
                    let text = format!("{}", status.message);
                    egui::Area::new("status_toast".into())
                        .anchor(egui::Align2::RIGHT_TOP, [-10.0, 40.0])
                        .show(ctx, |ui| {
                            ui.group(|ui| {
                                ui.colored_label(color, text);
                            });
                        });
                }
            }
        }
    }
}

fn main() -> Result<(), eframe::Error> {
    let mut world_override: Option<PathBuf> = None;
    let mut args = env::args().skip(1).peekable();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--world" | "-w" => {
                if let Some(path) = args.next() {
                    world_override = Some(PathBuf::from(path));
                } else {
                    eprintln!("--world requires a path argument");
                    return Ok(());
                }
            }
            "--help" | "-h" => {
                println!("Usage: zork-ui-egui [--world path]");
                return Ok(());
            }
            _ => {}
        }
    }

    let config = EngineConfig {
        data_path: None,
        media_enabled: true,
    };
    let _state = init_engine(config);

    // Try to load manifest (optional)
    let manifest = load_asset_manifest(Path::new("../assets/manifest.json")).ok();
    let manifest_status = match manifest.as_ref() {
        Some(m) => format!(
            "Chapter: {} | rooms {} | objects {}",
            game_display(m.game.as_deref()),
            m.rooms.len(),
            m.objects.len()
        ),
        None => "Manifest load failed or missing".to_string(),
    };

    // Try to load hero/thumb + pano placeholders from manifest (WEST-OF-HOUSE if exists)
    let (hero_image, pano_frames, placeholder_room) =
        load_scene_placeholders(manifest.as_ref(), Path::new("../assets/manifest.json"));
    let tts_audio = OutputStream::try_default().ok();
    let (pano_tx, pano_rx) = mpsc::channel();

    // Load parser + world to support intent display later
    let mut parser = load_parser_data(&ParserFiles::with_defaults()).ok();
    let mut parse_ctx = ParseContext {
        room: None,
        inventory: vec![],
    };
    let mut engine = init_engine(EngineConfig {
        data_path: None,
        media_enabled: true,
    });
    if let Some(p) = parser.as_mut() {
        let default_world = PathBuf::from("../out/zork1_world.json");
        let world_path = world_override.as_ref().unwrap_or(&default_world);
        let world_checksum = file_sha256(world_path).ok();
        if let Ok(world) = load_world(world_path) {
            let start_room = world
                .builder_meta
                .as_ref()
                .and_then(|m| m.start_room.clone())
                .or_else(|| {
                    world
                        .rooms
                        .keys()
                        .find(|k| k.eq_ignore_ascii_case("WEST-OF-HOUSE"))
                        .cloned()
                })
                .or_else(|| world.rooms.keys().next().cloned());
            parse_ctx.room = start_room.map(|r| r.to_uppercase());
            parse_ctx.inventory = world
                .objects
                .values()
                .filter(|o| {
                    o.location
                        .as_deref()
                        .map(|loc| {
                            loc.eq_ignore_ascii_case("INVENTORY")
                                || loc.eq_ignore_ascii_case("PLAYER")
                        })
                        .unwrap_or(false)
                })
                .map(|o| o.id.to_uppercase())
                .collect();
            parse_ctx.inventory.sort();
            parse_ctx.inventory.dedup();
            seed_actor_runtime(&world, &mut engine);
            let map_path = canonical_map_path(Path::new(".."), &world.game);
            p.canonical_map =
                load_canonical_map(&map_path, Some(&world.game), world_checksum.as_deref()).ok();
            p.world = Some(world.clone());
            p.world_index = Some(zork_core::WorldIndex::from_world(&world));
            p.world_vocab = build_world_vocab(&world);
        } else {
            eprintln!(
                "World file not found ({}); skipping context load.",
                world_path.display()
            );
        }
    }

    let font_dirs = candidate_font_dirs();
    let fonts = discover_fonts(&font_dirs);
    let default_regular = choose_font(
        &fonts,
        &["unscii-16-full", "notosansmono-regular", "regular"],
    );
    let default_bold = choose_font(&fonts, &["unscii-8-fantasy", "notosansmono-bold", "bold"]);

    let settings_path = PathBuf::from("../out/egui_settings.json");
    let loaded_settings = load_settings(
        &settings_path,
        default_regular.clone(),
        default_bold.clone(),
        "../out/egui_save.json".into(),
    );

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([900.0, 640.0]),
        ..Default::default()
    };
    eframe::run_native(
        "Zork UI (Rusty Knife)",
        options,
        Box::new(move |cc| {
            let hero_texture = hero_image.map(|(name, img)| {
                cc.egui_ctx
                    .load_texture(name, img, egui::TextureOptions::LINEAR)
            });
            let pano_frames = pano_frames.map(Arc::new);
            let (pano_texture, pano_size, pano_frames, pano_frame_index, pano_next_switch) =
                if let Some(frames) = pano_frames.clone() {
                    let first = frames
                        .frames
                        .first()
                        .expect("pano_frames should have at least one frame");
                    let tex = cc.egui_ctx.load_texture(
                        frames.name.clone(),
                        first.image.clone(),
                        egui::TextureOptions::LINEAR,
                    );
                    (
                        Some(tex),
                        Some(first.image.size),
                        Some(frames),
                        0usize,
                        0.0f64,
                    )
                } else {
                    (None, None, None, 0usize, 0.0f64)
                };
            let mut pano_cache = HashMap::new();
            let mut pano_cache_hot_order = VecDeque::new();
            let pano_cache_prefetch_order = VecDeque::new();
            let mut pano_cache_bytes = 0usize;
            let mut pano_rel_field = None;
            if let (Some(room_id), Some(frames)) = (placeholder_room.clone(), pano_frames.clone()) {
                if let Some(first) = frames.frames.first() {
                    pano_rel_field = Some(frames.name.clone());
                    let key = room_id.to_uppercase();
                    let approx_bytes = estimate_frames_bytes(&frames);
                    pano_cache.insert(
                        key.clone(),
                        PanoCacheEntry {
                            name: frames.name.clone(),
                            first_frame: first.clone(),
                            frames: Some(frames.clone()),
                            size: first.image.size,
                            loading: false,
                            prefetch: false,
                            approx_bytes,
                        },
                    );
                    pano_cache_hot_order.push_back(key);
                    pano_cache_bytes = approx_bytes;
                }
            }
            let (tts_stream, tts_handle) = match tts_audio {
                Some((stream, handle)) => (Some(stream), Some(handle)),
                None => (None, None),
            };
            let mut app = ZorkEguiApp {
                manifest_status,
                manifest: manifest.clone(),
                assets_base: PathBuf::from("../assets"),
                parser,
                ctx: parse_ctx,
                input: String::new(),
                input_has_focus: false,
                log: vec![LogEntry::Plain("(Rusty Knife UI ready)".into())],
                current_dialog: None,
                hero_texture,
                pano_texture,
                pano_size,
                pano_frames,
                pano_frame_index,
                pano_next_switch,
                pano_rel: pano_rel_field,
                pano_cache,
                pano_cache_hot_order,
                pano_cache_prefetch_order,
                pano_cache_bytes,
                pano_jobs: Vec::new(),
                pano_inflight: None,
                pano_job_started: HashMap::new(),
                pano_job_attempts: HashMap::new(),
                pano_generation: 0,
                pano_loader_rx: pano_rx,
                pano_loader_tx: pano_tx.clone(),
                scene_mode: loaded_settings.scene_mode,
                scene_yaw_deg: DEFAULT_YAW_DEG,
                scene_yaw_target_deg: DEFAULT_YAW_DEG,
                scene_pitch_deg: 0.0,
                scene_pitch_target_deg: 0.0,
                scene_invert_y: loaded_settings.scene_invert_y,
                scene_drag_sensitivity: loaded_settings.scene_drag_sensitivity,
                scene_fov_y_deg: loaded_settings.scene_fov_y_deg,
                show_options: false,
                status: None,
                status_time: None,
                current_time: 0.0,
                save_path: loaded_settings.save_path.clone(),
                builder_world_path: world_override.clone(),
                builder_log_path: world_override
                    .as_ref()
                    .map(|p| p.with_extension("log.jsonl")),
                show_help_panel: false,
                rebuild_rx: None,
                engine,
                map_zoom: 1.0,
                map_pan: egui::Vec2::ZERO,
                show_inventory: false,
                inventory_hover: None,
                show_hints: loaded_settings.show_hints,
                fonts: fonts.clone(),
                font_regular: loaded_settings
                    .font_regular
                    .clone()
                    .or(default_regular.clone()),
                font_bold: loaded_settings
                    .font_bold
                    .clone()
                    .or(default_bold.or(default_regular.clone())),
                font_size_base: loaded_settings.font_size_base,
                font_size_bold: loaded_settings.font_size_bold,
                font_error: None,
                tts_mode: loaded_settings.tts_mode,
                tts_volume: loaded_settings.tts_volume,
                palette: loaded_settings.palette,
                last_move_dir: None,
                tts_played_rooms: HashSet::new(),
                _tts_stream: tts_stream,
                tts_handle,
                tts_cancel: None,
                tts_sink: None,
                speech_auto_send: loaded_settings.speech_auto_send,
                speech_model_path: loaded_settings.speech_model_path.clone(),
                speech_language: loaded_settings.speech_language.clone(),
                speech_hotkey: loaded_settings.speech_hotkey.clone(),
                speech_enabled: loaded_settings.speech_enabled && SPEECH_BUILD_ENABLED,
                speech_state: SpeechUiState::Idle,
                speech_hotkey_down: false,
                speech_tx: None,
                speech_rx: None,
                speech_started_at: None,
                settings_path: settings_path.clone(),
                show_system_messages: loaded_settings.show_system_messages,
                pulse_next: 0.0,
                pulse_interval: 8.0,
                pulse_duration: 0.66,
                pulse_spans: Vec::new(),
                pulse_text: None,
            };
            app.apply_fonts(&cc.egui_ctx);
            if app.ctx.room.is_some() {
                app.reload_room_media(&cc.egui_ctx);
                app.log_room_description(true, false, true);
            }
            Box::new(app)
        }),
    )
}

fn load_scene_placeholders(
    manifest: Option<&AssetManifest>,
    manifest_path: &Path,
) -> (
    Option<(String, ColorImage)>,
    Option<AnimatedFrames>,
    Option<String>,
) {
    let manifest_owned;
    let manifest = match manifest {
        Some(m) => m,
        None => {
            manifest_owned = match load_asset_manifest(manifest_path) {
                Ok(m) => m,
                Err(_) => return (None, None, None),
            };
            &manifest_owned
        }
    };
    let base_dir = manifest_path
        .parent()
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| PathBuf::from("."));
    let room_id = manifest
        .rooms
        .get("WEST-OF-HOUSE")
        .map(|_| "WEST-OF-HOUSE")
        .or_else(|| manifest.rooms.keys().next().map(|k| k.as_str()))
        .unwrap_or("WEST-OF-HOUSE");
    let entry = resolve_room_assets(manifest, room_id, &base_dir);

    let hero_rel = entry
        .hero
        .as_ref()
        .or_else(|| entry.thumb.as_ref())
        .or_else(|| entry.ambient.as_ref());
    let pano_rel = entry.pano.as_ref();

    let hero_img = hero_rel.and_then(|rel| load_static_image(&base_dir, rel));
    let pano_frames = pano_rel.and_then(|rel| load_pano_frames(&base_dir, rel));
    (hero_img, pano_frames, Some(room_id.to_string()))
}

fn load_static_image(base: &Path, rel: &str) -> Option<(String, ColorImage)> {
    let full_path = base.join(rel);
    let bytes = std::fs::read(full_path).ok()?;
    let dyn_img = image::load_from_memory(&bytes).ok()?.to_rgba8();
    let (w, h) = dyn_img.dimensions();
    let color_image =
        egui::ColorImage::from_rgba_unmultiplied([w as usize, h as usize], dyn_img.as_raw());
    Some((rel.to_string(), color_image))
}

fn load_pano_frames(base: &Path, rel: &str) -> Option<AnimatedFrames> {
    let full_path = base.join(rel);
    let ext = full_path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();
    if ext == "webp" {
        let bytes = std::fs::read(&full_path).ok()?;
        if let Some(frames) = load_webp_frames_from_bytes(&bytes) {
            return Some(AnimatedFrames {
                name: rel.to_string(),
                frames,
            });
        }
    }
    // Fallback: treat as static image frame
    load_static_image(base, rel).map(|(name, mut img)| {
        apply_pano_seam_blend(&mut img, PANO_SEAM_BLEND_PX);
        AnimatedFrames {
            name,
            frames: vec![AnimatedFrame {
                image: img,
                duration_secs: 1_000_000.0, // effectively static
            }],
        }
    })
}

fn load_webp_frames_from_bytes(bytes: &[u8]) -> Option<Vec<AnimatedFrame>> {
    if let Some(frames) = load_webp_frames_libwebp(bytes) {
        return Some(frames);
    }
    let cursor = Cursor::new(bytes);
    let decoder = image::codecs::webp::WebPDecoder::new(cursor).ok()?;
    let has_anim = decoder.has_animation();
    let frames_iter = decoder.into_frames();
    let mut out = Vec::new();
    for frame_res in frames_iter {
        let frame = frame_res.ok()?;
        let (numer, denom) = frame.delay().numer_denom_ms();
        let millis = if denom == 0 {
            0.0
        } else {
            numer as f32 / denom as f32
        };
        let secs = (millis / 1000.0).max(0.01);
        let buffer = frame.into_buffer();
        let (w, h) = buffer.dimensions();
        let img =
            egui::ColorImage::from_rgba_unmultiplied([w as usize, h as usize], buffer.as_raw());
        out.push(AnimatedFrame {
            image: img,
            duration_secs: if has_anim { secs } else { 1_000_000.0 },
        });
    }
    if out.is_empty() {
        None
    } else {
        apply_pano_seam_blend_frames(&mut out);
        Some(out)
    }
}

fn load_webp_frames_libwebp(bytes: &[u8]) -> Option<Vec<AnimatedFrame>> {
    let decoder = WebpAnimDecoder::new(bytes).ok()?;
    let dims = decoder.dimensions();
    let frames: Vec<_> = decoder.into_iter().collect();
    if frames.is_empty() {
        return None;
    }
    let mut out = Vec::new();
    for (idx, frame) in frames.iter().enumerate() {
        let next_ts = frames
            .get(idx + 1)
            .map(|f| f.timestamp())
            .unwrap_or_else(|| frame.timestamp() + 40);
        let dur_ms = (next_ts - frame.timestamp()).max(10);
        let dur_secs = (dur_ms as f32) / 1000.0;
        out.push(AnimatedFrame {
            image: egui::ColorImage::from_rgba_unmultiplied(
                [dims.0 as usize, dims.1 as usize],
                frame.data(),
            ),
            duration_secs: dur_secs,
        });
    }
    apply_pano_seam_blend_frames(&mut out);
    Some(out)
}

#[cfg(feature = "mp4-video")]
fn decode_mp4_first_frame(path: &Path) -> Option<(AnimatedFrame, [usize; 2], bool)> {
    let _ = ffmpeg_next::init();
    let path_buf = path.to_path_buf();
    let mut ictx = ffmpeg_next::format::input(&path_buf).ok()?;
    let input = ictx.streams().best(ffmpeg_next::media::Type::Video)?;
    let video_index = input.index();
    let mut decoder = ffmpeg_next::codec::context::Context::from_parameters(input.parameters())
        .ok()?
        .decoder()
        .video()
        .ok()?;
    let mut scaler = ffmpeg_next::software::scaling::Context::get(
        decoder.format(),
        decoder.width(),
        decoder.height(),
        ffmpeg_next::format::Pixel::RGBA,
        decoder.width(),
        decoder.height(),
        ffmpeg_next::software::scaling::Flags::BILINEAR,
    )
    .ok()?;
    for (stream, packet) in ictx.packets() {
        if stream.index() != video_index {
            continue;
        }
        decoder.send_packet(&packet).ok()?;
        let mut frame = ffmpeg_next::util::frame::Video::empty();
        if decoder.receive_frame(&mut frame).is_ok() {
            let mut rgba = ffmpeg_next::util::frame::Video::empty();
            scaler.run(&frame, &mut rgba).ok()?;
            let w = rgba.width() as usize;
            let h = rgba.height() as usize;
            let stride = rgba.stride(0);
            let data = rgba.data(0);
            let mut out = Vec::with_capacity(w * h * 4);
            for y in 0..h {
                let start = y * stride as usize;
                out.extend_from_slice(&data[start..start + w * 4]);
            }
            let mut img = egui::ColorImage::from_rgba_unmultiplied([w, h], &out);
            apply_pano_seam_blend(&mut img, PANO_SEAM_BLEND_PX);
            return Some((
                AnimatedFrame {
                    image: img,
                    duration_secs: 1.0 / 30.0,
                },
                [w, h],
                true,
            ));
        }
    }
    None
}

#[cfg(not(feature = "mp4-video"))]
fn decode_mp4_first_frame(_path: &Path) -> Option<(AnimatedFrame, [usize; 2], bool)> {
    None
}

#[cfg(feature = "mp4-video")]
fn decode_mp4_frames(path: &Path) -> Option<Vec<AnimatedFrame>> {
    let _ = ffmpeg_next::init();
    let path_buf = path.to_path_buf();
    let mut ictx = ffmpeg_next::format::input(&path_buf).ok()?;
    let input = ictx.streams().best(ffmpeg_next::media::Type::Video)?;
    let video_index = input.index();
    let mut decoder = ffmpeg_next::codec::context::Context::from_parameters(input.parameters())
        .ok()?
        .decoder()
        .video()
        .ok()?;
    let mut scaler = ffmpeg_next::software::scaling::Context::get(
        decoder.format(),
        decoder.width(),
        decoder.height(),
        ffmpeg_next::format::Pixel::RGBA,
        decoder.width(),
        decoder.height(),
        ffmpeg_next::software::scaling::Flags::BILINEAR,
    )
    .ok()?;
    let time_base = input.time_base();
    let avg_rate = input.avg_frame_rate();
    let default_dur = if avg_rate.numerator() > 0 && avg_rate.denominator() > 0 {
        (avg_rate.denominator() as f32 / avg_rate.numerator() as f32 * 4.0).max(0.01)
    } else {
        2.0 / 30.0
    };
    let speed_scale = 4.0;
    let mut out = Vec::new();
    let mut last_ts = None;
    for (stream, packet) in ictx.packets() {
        if stream.index() != video_index {
            continue;
        }
        decoder.send_packet(&packet).ok()?;
        loop {
            let mut frame = ffmpeg_next::util::frame::Video::empty();
            match decoder.receive_frame(&mut frame) {
                Ok(_) => {
                    let mut rgba = ffmpeg_next::util::frame::Video::empty();
                    scaler.run(&frame, &mut rgba).ok()?;
                    let w = rgba.width() as usize;
                    let h = rgba.height() as usize;
                    let stride = rgba.stride(0);
                    let data = rgba.data(0);
                    let mut buf = Vec::with_capacity(w * h * 4);
                    for y in 0..h {
                        let start = y * stride as usize;
                        buf.extend_from_slice(&data[start..start + w * 4]);
                    }
                    let mut img = egui::ColorImage::from_rgba_unmultiplied([w, h], &buf);
                    apply_pano_seam_blend(&mut img, PANO_SEAM_BLEND_PX);
                    let ts = frame.timestamp().unwrap_or(0);
                    let ts_sec = ts as f64 * time_base.0 as f64 / time_base.1 as f64;
                    let dur = if let Some(prev) = last_ts {
                        (((ts_sec - prev as f64).max(0.01)) * speed_scale as f64) as f32
                    } else {
                        default_dur
                    };
                    last_ts = Some(ts);
                    out.push(AnimatedFrame {
                        image: img,
                        duration_secs: dur,
                    });
                }
                Err(ffmpeg_next::Error::Other { errno })
                    if errno == ffmpeg_next::util::error::EAGAIN =>
                {
                    break
                }
                Err(ffmpeg_next::Error::Eof) => break,
                Err(_) => break,
            }
        }
    }
    // drain remaining
    decoder.send_eof().ok()?;
    loop {
        let mut frame = ffmpeg_next::util::frame::Video::empty();
        match decoder.receive_frame(&mut frame) {
            Ok(_) => {
                let mut rgba = ffmpeg_next::util::frame::Video::empty();
                scaler.run(&frame, &mut rgba).ok()?;
                let w = rgba.width() as usize;
                let h = rgba.height() as usize;
                let stride = rgba.stride(0);
                let data = rgba.data(0);
                let mut buf = Vec::with_capacity(w * h * 4);
                for y in 0..h {
                    let start = y * stride as usize;
                    buf.extend_from_slice(&data[start..start + w * 4]);
                }
                let mut img = egui::ColorImage::from_rgba_unmultiplied([w, h], &buf);
                apply_pano_seam_blend(&mut img, PANO_SEAM_BLEND_PX);
                let dur = default_dur;
                out.push(AnimatedFrame {
                    image: img,
                    duration_secs: dur,
                });
            }
            Err(ffmpeg_next::Error::Other { errno })
                if errno == ffmpeg_next::util::error::EAGAIN =>
            {
                break
            }
            Err(ffmpeg_next::Error::Eof) => break,
            Err(_) => break,
        }
    }
    if out.is_empty() {
        None
    } else {
        Some(out)
    }
}

#[cfg(not(feature = "mp4-video"))]
fn decode_mp4_frames(_path: &Path) -> Option<Vec<AnimatedFrame>> {
    None
}

fn load_pano_first_frame(
    base: &Path,
    rel: &str,
) -> Option<(AnimatedFrame, [usize; 2], Option<PanoSource>, bool)> {
    let full_path = base.join(rel);
    let ext = full_path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();
    if ext == "mp4" {
        if let Some((frame, size, has_anim)) = decode_mp4_first_frame(&full_path) {
            return Some((frame, size, Some(PanoSource::Mp4Path(full_path)), has_anim));
        }
    }
    if ext == "webp" {
        let bytes = Arc::new(std::fs::read(&full_path).ok()?);
        let cursor = Cursor::new(bytes.as_slice());
        let decoder = image::codecs::webp::WebPDecoder::new(cursor).ok()?;
        let has_anim = decoder.has_animation();
        let mut frames_iter = decoder.into_frames();
        let first = frames_iter.next()?.ok()?;
        drop(frames_iter);
        let (numer, denom) = first.delay().numer_denom_ms();
        let millis = if denom == 0 {
            0.0
        } else {
            numer as f32 / denom as f32
        };
        let secs = (millis / 1000.0).max(0.01);
        let buffer = first.into_buffer();
        let (w, h) = buffer.dimensions();
        let img =
            egui::ColorImage::from_rgba_unmultiplied([w as usize, h as usize], buffer.as_raw());
        let frame = AnimatedFrame {
            image: img,
            duration_secs: if has_anim { secs } else { 1_000_000.0 },
        };
        return Some((
            frame,
            [w as usize, h as usize],
            Some(PanoSource::WebpBytes(bytes.clone())),
            has_anim,
        ));
    }
    // Fallback: treat as static image frame
    load_static_image(base, rel).map(|(_, img)| {
        let size = img.size;
        (
            AnimatedFrame {
                image: img,
                duration_secs: 1_000_000.0,
            },
            size,
            None,
            false,
        )
    })
}

impl ZorkEguiApp {
    fn prettify_id(&self, raw: &str) -> String {
        if let Some(parser) = self.parser.as_ref() {
            if let Some(world) = parser.world.as_ref() {
                if let Some(obj) = world.objects.get(raw) {
                    if let Some(name) = obj.names.first() {
                        return title_case(name);
                    }
                }
                if let Some(room) = world.rooms.get(raw) {
                    if let Some(desc) = room.desc.as_ref() {
                        if !desc.is_empty() {
                            return desc.trim_matches('"').to_string();
                        }
                    }
                }
            }
        }
        title_case(raw)
    }

    fn prettify_exit(&self, dir: &str) -> String {
        match dir.to_uppercase().as_str() {
            "N" | "NORTH" => "North".into(),
            "S" | "SOUTH" => "South".into(),
            "E" | "EAST" => "East".into(),
            "W" | "WEST" => "West".into(),
            "NE" | "NORTHEAST" => "Northeast".into(),
            "NW" | "NORTHWEST" => "Northwest".into(),
            "SE" | "SOUTHEAST" => "Southeast".into(),
            "SW" | "SOUTHWEST" => "Southwest".into(),
            "U" | "UP" => "Up".into(),
            "D" | "DOWN" => "Down".into(),
            "IN" | "ENTER" => "In".into(),
            "OUT" | "EXIT" => "Out".into(),
            other => title_case(other),
        }
    }

    fn refresh_world_views(&mut self) {
        if let Some(parser) = self.parser.as_mut() {
            if let Some(world) = parser.world.as_ref() {
                parser.world_index = Some(zork_core::WorldIndex::from_world(world));
                parser.world_vocab = build_world_vocab(world);
            }
        }
    }

    fn handle_dialog_view(&mut self, view: DialogView) {
        self.log_dialog_view(&view);
        if view.ended {
            self.current_dialog = None;
            self.engine.dialog = None;
        } else {
            self.current_dialog = Some(view);
        }
    }

    fn start_dialog_with(&mut self, npc_id: &str, node_id: Option<&str>) {
        let result = {
            let parser = match self.parser.as_mut() {
                Some(p) => p,
                None => {
                    return self.push_plain("(dialog) no parser loaded; use /newworld first");
                }
            };
            if parser.world_index.is_none() {
                if let Some(world) = parser.world.as_ref() {
                    parser.world_index = Some(zork_core::WorldIndex::from_world(world));
                    parser.world_vocab = build_world_vocab(world);
                }
            }
            match (parser.world.as_mut(), parser.world_index.as_mut()) {
                (Some(world), Some(idx)) => {
                    start_dialog(world, idx, &mut self.ctx, &mut self.engine, npc_id, node_id)
                }
                _ => Err("(dialog) no world loaded; use /newworld first".into()),
            }
        };
        match result {
            Ok(view) => {
                self.handle_dialog_view(view);
                self.input_has_focus = true;
            }
            Err(e) => self.push_plain(format!("(dialog) {}", e)),
        }
    }

    fn choose_dialog_option_idx(&mut self, choice_index: usize) {
        let result = {
            let parser = match self.parser.as_mut() {
                Some(p) => p,
                None => {
                    return self.push_plain("(dialog) no parser loaded; use /newworld first");
                }
            };
            if parser.world_index.is_none() {
                if let Some(world) = parser.world.as_ref() {
                    parser.world_index = Some(zork_core::WorldIndex::from_world(world));
                    parser.world_vocab = build_world_vocab(world);
                }
            }
            match (parser.world.as_mut(), parser.world_index.as_mut()) {
                (Some(world), Some(idx)) => {
                    choose_dialog_option(world, idx, &mut self.ctx, &mut self.engine, choice_index)
                }
                _ => Err("(dialog) no world loaded; use /newworld first".into()),
            }
        };
        match result {
            Ok(view) => {
                self.handle_dialog_view(view);
                self.input_has_focus = true;
            }
            Err(e) => self.push_plain(format!("(dialog) {}", e)),
        }
    }

    fn render_builder_help(&self, ui: &mut egui::Ui) {
        ui.group(|ui| {
            let pal = self.current_palette();
            ui.heading("Builder Commands");
            let rows = [
                (
                    "/newworld [name|path]",
                    "create a blank world (writes to out/worlds/ by default)",
                ),
                ("/name <new_room_id>", "rename current room"),
                ("/carve <dir>", "add an exit (creates stub + reciprocal)"),
                (
                    "/exit <dir> <dest> [guard] [message] [oneway]",
                    "guarded/one-way exit without stubs",
                ),
                (
                    "/door <dir> [dest] <locked|unlocked> [flag] [key]",
                    "door object + exits (locks gate travel)",
                ),
                (
                    "/ldesc <text>",
                    "set long description (also updates short name if default)",
                ),
                ("/sdesc <text>", "set short description"),
                ("/note <text>", "add builder note to the room"),
                ("/ambient <text>", "add ambient/unconditional flavor text"),
                ("/variant <flag|none> <text>", "conditional description fragment"),
                ("/start <room>", "set start room metadata"),
                ("/nobj <name>, <desc>[, container]", "add object to current room"),
                ("/objname <name>, <newname>", "rename an object id/name list"),
                ("/moveobj <obj> <room|inventory|object>", "move an object"),
                ("/objflags <obj> +FLAG -FLAG ...", "toggle object flags"),
                ("/flag [+]FLAG|-FLAG [...]", "set/clear global flags"),
                ("/cloneobj <src> <new> [dest]", "clone an object"),
                ("/cloneroom <src> <new>", "clone a room template (no exits)"),
                ("/exportroom <room> <path>", "write a room JSON snippet"),
                ("/importr <path> [new_id]", "import a room template (clears exits)"),
                (
                    "/action <cmd> <flags_required> <flags_changed>",
                    "add room action",
                ),
                (
                    "/counter <name> <value|+N|-N> [drain N]",
                    "set/mutate counters (health/O2/etc.)",
                ),
                (
                    "/npc <name>, <desc>[, room]",
                    "create/update NPC in this room",
                ),
                (
                    "/npcnode <npc> <node> \"text\" [requires] [effects flags/counters/items/credits] [repeat]",
                    "add/replace dialog node",
                ),
                (
                    "/npclink <npc> <from> <to> \"prompt\" [requires] [effects flags/counters/items/credits] [hidden]",
                    "add/replace dialog link",
                ),
                ("/npcplace <npc> [room]", "move NPC"),
                ("/npcsay <npc> [node] | /npcchoose <idx>", "run dialog"),
                (
                    "/lint",
                    "severity-tagged desc/exit/NPC/placement/asset checks",
                ),
                ("/saveworld [path]", "save world snapshot (and append log)"),
                (
                    "/rebuild",
                    "save, rebuild parser vocab, regenerate canonical map",
                ),
            ];
            for (cmd, desc) in rows {
                ui.horizontal_wrapped(|ui| {
                    ui.colored_label(pal.fg_contrast, cmd);
                    ui.add(
                        egui::Label::new(egui::RichText::new(desc).color(pal.fg)).wrap(true),
                    );
                });
            }
            ui.separator();
            ui.colored_label(
                pal.fg,
                "Flags: list required flags separated by space/commas; changes accept FLAG to set, !FLAG or -FLAG to clear.",
            );
            ui.colored_label(
                pal.fg,
                "Builder actions fire before normal parser; flags map to EngineState.global_state.flags.",
            );
            ui.colored_label(
                pal.fg,
                "Paths: world/log default to out/worlds/<name>.json/.log.jsonl unless overridden in /newworld or /saveworld.",
            );
        });
    }

    fn persist_builder_change(&mut self, change: Option<&BuilderChange>) {
        if let Some(world) = self.parser.as_ref().and_then(|p| p.world.as_ref()) {
            if let Some(path) = self.builder_world_path.as_ref() {
                if let Some(dir) = path.parent() {
                    let _ = fs::create_dir_all(dir);
                }
                if let Err(e) = save_world(path, world) {
                    self.push_plain(format!("(builder) save failed: {}", e));
                }
            }
        }
        if let (Some(ch), Some(log_path)) = (change, self.builder_log_path.as_ref()) {
            if let Some(dir) = log_path.parent() {
                let _ = fs::create_dir_all(dir);
            }
            if let Err(e) = append_builder_change(log_path, ch) {
                self.push_plain(format!("(builder) log append failed: {}", e));
            }
        }
    }

    fn prettify_sentence(&self, text: &str) -> String {
        let parser = if let Some(p) = self.parser.as_ref() {
            p
        } else {
            return text.to_string();
        };
        let world = if let Some(w) = parser.world.as_ref() {
            w
        } else {
            return text.to_string();
        };
        let mut out = String::new();
        for (i, raw_word) in text.split_whitespace().enumerate() {
            if i > 0 {
                out.push(' ');
            }
            let (prefix, core, suffix) = split_word(raw_word);
            let upper = core.to_uppercase();
            let pretty = if world.rooms.contains_key(&upper) || world.objects.contains_key(&upper) {
                self.prettify_id(&upper)
            } else {
                core.to_string()
            };
            out.push_str(prefix);
            out.push_str(&pretty);
            out.push_str(suffix);
        }
        out
    }

    fn render_scene(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        ui.label("Scene");
        match self.scene_mode {
            SceneMode::Off => {
                ui.colored_label(egui::Color32::GRAY, "Scene disabled");
            }
            SceneMode::ImageOnly => {
                self.render_hero_scene(ui);
            }
            SceneMode::Auto => {
                if self.pano_texture.is_some() {
                    self.render_pano_scene(ui, ctx);
                } else {
                    self.render_hero_scene(ui);
                }
            }
        }
        // Fill the group to keep layout consistent with the context pane height.
        ui.allocate_space(ui.available_size());
    }

    fn render_hero_scene(&mut self, ui: &mut egui::Ui) {
        if let Some(tex) = &self.hero_texture {
            let available = ui.available_size();
            let img_w = available.x.min(520.0).max(160.0);
            let mut img_h = img_w * 0.6;
            let max_h = (available.y - 16.0).max(140.0);
            if img_h > max_h {
                img_h = max_h;
            }
            let size = egui::vec2(img_w, img_h);
            ui.image(egui::ImageSource::Texture((tex.id(), size).into()));
        } else {
            ui.colored_label(egui::Color32::GRAY, "No scene image loaded");
        }
    }

    fn render_pano_scene(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        let tex_id = if let Some(tex) = self.pano_texture.as_ref() {
            tex.id()
        } else {
            self.render_hero_scene(ui);
            return;
        };

        let available = ui.available_size();
        let mut img_w = available.x.max(220.0);
        let aspect = self
            .pano_size
            .map(|s| (s[0] as f32 / s[1] as f32).max(0.1))
            .unwrap_or(2.0);
        let mut img_h = (img_w / aspect).max(160.0);
        let max_h = (available.y - 12.0).max(160.0);
        if img_h > max_h {
            img_h = max_h;
            img_w = img_h * aspect;
        }
        let desired = egui::vec2(img_w, img_h);
        let (rect, response) = ui.allocate_exact_size(desired, egui::Sense::drag());

        self.handle_pano_zoom(&response, ctx);
        self.handle_pano_drag(&response, ctx);
        self.update_pano_animation(ctx);

        let fov_y = self.scene_fov_y_deg.clamp(30.0, 110.0);
        let mesh = build_pano_mesh(
            rect,
            tex_id,
            self.scene_yaw_deg,
            self.scene_pitch_deg,
            fov_y,
        );
        let painter = ui.painter_at(rect);
        painter.add(mesh);

        if response.hovered() {
            let overlay = format!(
                "Yaw {:.0} deg, Pitch {:.0} deg, FOV {:.0} deg{}",
                self.scene_yaw_deg,
                self.scene_pitch_deg,
                self.scene_fov_y_deg,
                if self.scene_invert_y { " (inv Y)" } else { "" }
            );
            let pal = self.current_palette();
            painter.text(
                rect.left_top() + egui::vec2(8.0, 8.0),
                egui::Align2::LEFT_TOP,
                overlay,
                egui::TextStyle::Small.resolve(ui.style()),
                pal.fg,
            );
            painter.text(
                rect.left_bottom() + egui::vec2(8.0, -8.0),
                egui::Align2::LEFT_BOTTOM,
                "Drag to pan | Wheel to zoom | Double-click to reset",
                egui::TextStyle::Small.resolve(ui.style()),
                pal.fg,
            );
        }

        if self.is_pano_loading() {
            self.render_pano_loading_indicator(&painter, rect);
        }
    }

    fn toggle_inventory_overlay(&mut self) {
        self.show_inventory = !self.show_inventory;
        if !self.show_inventory {
            self.inventory_hover = None;
        }
    }

    fn is_pano_loading(&self) -> bool {
        let key = match self.ctx.room.as_ref() {
            Some(r) => r.to_uppercase(),
            None => return false,
        };
        self.pano_cache
            .get(&key)
            .map(|e| e.loading && e.frames.is_none())
            .unwrap_or(false)
    }

    fn render_pano_loading_indicator(&self, painter: &egui::Painter, rect: egui::Rect) {
        let h = 4.0;
        let y0 = rect.max.y - h;
        let base = self.palette.data().fg;
        let [r, g, b, _] = base.to_array();
        let bg = Color32::from_rgba_unmultiplied(r, g, b, 60);
        let fg = Color32::from_rgba_unmultiplied(r, g, b, 200);
        painter.rect_filled(
            egui::Rect::from_min_max(
                egui::pos2(rect.min.x, y0),
                egui::pos2(rect.max.x, rect.max.y),
            ),
            0.0,
            bg,
        );
        let w = rect.width();
        let t = (self.current_time * 0.6) % 1.0;
        let span = w * 0.35;
        let start = rect.min.x + t as f32 * w;
        let end = (start + span).min(rect.max.x);
        let min_x = start.max(rect.min.x);
        if end > min_x {
            painter.rect_filled(
                egui::Rect::from_min_max(egui::pos2(min_x, y0), egui::pos2(end, rect.max.y)),
                0.0,
                fg,
            );
        }
    }

    fn inventory_entries(&self) -> Vec<InventoryEntry> {
        self.ctx
            .inventory
            .iter()
            .map(|id| InventoryEntry {
                id: id.clone(),
                label: self.prettify_id(id),
                can_open: self.can_open_inventory_object(id),
            })
            .collect()
    }

    fn can_open_inventory_object(&self, obj_id: &str) -> bool {
        let parser = match self.parser.as_ref() {
            Some(p) => p,
            None => return false,
        };
        let world = match parser.world.as_ref() {
            Some(w) => w,
            None => return false,
        };
        let obj = match world.objects.get(&obj_id.to_uppercase()) {
            Some(o) => o,
            None => return false,
        };
        let has_flag = |flag: &str| obj.flags.iter().any(|f| f.eq_ignore_ascii_case(flag));
        if has_flag("CONTBIT") || has_flag("DOORBIT") || has_flag("OPENBIT") {
            return true;
        }
        obj.semantic_types
            .iter()
            .any(|t| t.eq_ignore_ascii_case("container") || t.eq_ignore_ascii_case("portal"))
    }

    fn render_inventory_overlay(
        &mut self,
        ui: &mut egui::Ui,
        available: egui::Vec2,
        ctx: &egui::Context,
    ) {
        ui.set_min_size(available);
        ui.horizontal(|ui| {
            ui.label(RichText::new("Inventory").underline());
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                if ui.button("Close Inv").clicked() {
                    self.toggle_inventory_overlay();
                }
            });
        });
        ui.add_space(8.0);
        let items = self.inventory_entries();
        if items.is_empty() {
            ui.label("You are empty-handed.");
            return;
        }
        let columns: usize = if available.x > 420.0 { 2 } else { 1 };
        let mut hovered: Option<String> = None;
        egui::ScrollArea::vertical()
            .auto_shrink([false; 2])
            .max_height(available.y)
            .show(ui, |ui| {
                egui::Grid::new("inventory_grid")
                    .num_columns(columns)
                    .spacing(egui::vec2(12.0, 12.0))
                    .min_col_width((available.x / columns as f32).max(120.0))
                    .show(ui, |ui| {
                        for (idx, item) in items.iter().enumerate() {
                            self.render_inventory_item(ui, item, &mut hovered, ctx);
                            if (idx + 1) % columns == 0 {
                                ui.end_row();
                            }
                        }
                        if items.len() % columns != 0 {
                            ui.end_row();
                        }
                    });
            });
        self.inventory_hover = hovered;
    }

    fn render_inventory_item(
        &mut self,
        ui: &mut egui::Ui,
        item: &InventoryEntry,
        hovered: &mut Option<String>,
        ctx: &egui::Context,
    ) {
        ui.vertical(|ui| {
            let label = format!("• {}", item.label);
            let resp = ui.add(egui::Label::new(label).sense(egui::Sense::hover()));
            let show_buttons =
                self.inventory_hover.as_deref() == Some(item.id.as_str()) || resp.hovered();
            if resp.hovered() {
                *hovered = Some(item.id.clone());
            }
            let mut buttons_hovered = false;
            if show_buttons {
                ui.add_space(4.0);
                ui.horizontal(|ui| {
                    let look_resp = ui.button("Look");
                    if look_resp.clicked() {
                        let target = self.prettify_id(&item.id);
                        self.send_command(&format!("look {}", target.to_lowercase()), ctx);
                    }
                    if look_resp.hovered() {
                        buttons_hovered = true;
                    }
                    if item.can_open {
                        let open_resp = ui.button("Open");
                        if open_resp.clicked() {
                            let target = self.prettify_id(&item.id);
                            self.send_command(&format!("open {}", target.to_lowercase()), ctx);
                        }
                        if open_resp.hovered() {
                            buttons_hovered = true;
                        }
                    }
                });
            }
            if buttons_hovered {
                *hovered = Some(item.id.clone());
            }
            if show_buttons
                && hovered.is_none()
                && self.inventory_hover.as_deref() == Some(item.id.as_str())
            {
                *hovered = Some(item.id.clone());
            }
        });
    }

    fn handle_pano_drag(&mut self, response: &egui::Response, ctx: &egui::Context) {
        if response.dragged() {
            let delta = ctx.input(|i| i.pointer.delta());
            let factor = self.scene_drag_sensitivity.max(0.01);
            self.scene_yaw_target_deg =
                (self.scene_yaw_target_deg + delta.x * factor).rem_euclid(360.0);
            let sign = if self.scene_invert_y { 1.0 } else { -1.0 };
            self.scene_pitch_target_deg = (self.scene_pitch_target_deg + sign * delta.y * factor)
                .clamp(-MAX_PITCH_DEG, MAX_PITCH_DEG);
        }
        if response.double_clicked() {
            self.scene_yaw_target_deg = DEFAULT_YAW_DEG;
            self.scene_pitch_target_deg = 0.0;
        }
    }

    fn handle_pano_zoom(&mut self, response: &egui::Response, ctx: &egui::Context) {
        if !response.hovered() {
            return;
        }
        let scroll = ctx.input(|i| i.smooth_scroll_delta.y);
        if scroll.abs() < f32::EPSILON {
            return;
        }
        // Scroll up to zoom in (narrower FOV), down to zoom out.
        let new_fov = (self.scene_fov_y_deg - scroll * 0.25).clamp(20.0, 110.0);
        self.scene_fov_y_deg = new_fov;
    }

    fn update_pano_animation(&mut self, ctx: &egui::Context) {
        let (Some(anim), Some(tex)) = (self.pano_frames.as_ref(), self.pano_texture.as_mut())
        else {
            return;
        };
        if anim.frames.len() <= 1 {
            return;
        }
        let now = ctx.input(|i| i.time);
        if self.pano_next_switch <= 0.0 {
            let dur = anim.frames[self.pano_frame_index].duration_secs.max(0.01) as f64;
            self.pano_next_switch = now + dur;
            ctx.request_repaint_after(std::time::Duration::from_secs_f64(dur));
            return;
        }
        if now + f64::EPSILON >= self.pano_next_switch {
            self.pano_frame_index = (self.pano_frame_index + 1) % anim.frames.len();
            let frame = &anim.frames[self.pano_frame_index];
            tex.set(frame.image.clone(), egui::TextureOptions::LINEAR);
            let dur = frame.duration_secs.max(0.01) as f64;
            self.pano_next_switch = now + dur;
            ctx.request_repaint_after(std::time::Duration::from_secs_f64(dur));
        } else {
            let wait = (self.pano_next_switch - now).max(0.0);
            ctx.request_repaint_after(std::time::Duration::from_secs_f64(wait));
        }
    }

    fn smooth_scene_angles(&mut self, ctx: &egui::Context) {
        // Lerp toward targets to soften jerky input.
        let alpha = 0.18;
        // Wrap yaw shortest path.
        let mut delta_yaw = self.scene_yaw_target_deg - self.scene_yaw_deg;
        delta_yaw = (delta_yaw + 540.0).rem_euclid(360.0) - 180.0;
        self.scene_yaw_deg = (self.scene_yaw_deg + delta_yaw * alpha).rem_euclid(360.0);

        let delta_pitch = (self.scene_pitch_target_deg - self.scene_pitch_deg).clamp(-180.0, 180.0);
        self.scene_pitch_deg =
            (self.scene_pitch_deg + delta_pitch * alpha).clamp(-MAX_PITCH_DEG, MAX_PITCH_DEG);

        // Keep rendering smooth when moving toward targets.
        ctx.request_repaint();
    }

    fn apply_pano_cache_entry(
        &mut self,
        ctx: &egui::Context,
        room_key: &str,
        entry: &PanoCacheEntry,
    ) {
        if let Some(e) = self.pano_cache.get_mut(room_key) {
            e.prefetch = false;
        }
        self.remove_from_pano_orders(room_key);
        self.pano_cache_hot_order.push_back(room_key.to_string());
        self.pano_size = Some(entry.size);
        self.pano_frames = entry.frames.clone();
        self.pano_frame_index = 0;
        self.pano_next_switch = 0.0;
        self.pano_rel = Some(entry.name.clone());
        if let Some(tex) = self.pano_texture.as_mut() {
            tex.set(
                entry.first_frame.image.clone(),
                egui::TextureOptions::LINEAR,
            );
        } else {
            let tex = ctx.load_texture(
                entry.name.clone(),
                entry.first_frame.image.clone(),
                egui::TextureOptions::LINEAR,
            );
            self.pano_texture = Some(tex);
        }
    }

    fn insert_pano_cache(&mut self, room_key: String, entry: PanoCacheEntry) {
        self.remove_from_pano_orders(&room_key);
        if let Some(old) = self.pano_cache.insert(room_key.clone(), entry) {
            self.pano_cache_bytes = self.pano_cache_bytes.saturating_sub(old.approx_bytes);
        }
        if let Some(entry) = self.pano_cache.get(&room_key) {
            self.pano_cache_bytes = self.pano_cache_bytes.saturating_add(entry.approx_bytes);
            if entry.prefetch {
                self.pano_cache_prefetch_order
                    .push_back(room_key.to_string());
            } else {
                self.pano_cache_hot_order.push_back(room_key.to_string());
            }
        }
        self.trim_pano_cache();
    }

    fn remove_from_pano_orders(&mut self, room_key: &str) {
        self.pano_cache_hot_order
            .retain(|k| !k.eq_ignore_ascii_case(room_key));
        self.pano_cache_prefetch_order
            .retain(|k| !k.eq_ignore_ascii_case(room_key));
    }

    fn trim_pano_cache(&mut self) {
        let mut total_limit = PANO_CACHE_HOT_LIMIT + PANO_CACHE_PREFETCH_LIMIT;
        if total_limit == 0 {
            total_limit = 1;
        }
        let evict_one = |prefetch_first: bool,
                         hot_order: &mut VecDeque<String>,
                         pref_order: &mut VecDeque<String>,
                         cache: &mut HashMap<String, PanoCacheEntry>,
                         bytes: &mut usize| {
            let victim = if prefetch_first {
                pref_order.pop_front().or_else(|| hot_order.pop_front())
            } else {
                hot_order.pop_front().or_else(|| pref_order.pop_front())
            };
            if let Some(key) = victim {
                if let Some(entry) = cache.remove(&key) {
                    *bytes = bytes.saturating_sub(entry.approx_bytes);
                }
            }
        };

        while self.pano_cache_hot_order.len() + self.pano_cache_prefetch_order.len() > total_limit {
            evict_one(
                true,
                &mut self.pano_cache_hot_order,
                &mut self.pano_cache_prefetch_order,
                &mut self.pano_cache,
                &mut self.pano_cache_bytes,
            );
        }
        while self.pano_cache_bytes > PANO_MEM_BUDGET_BYTES {
            evict_one(
                true,
                &mut self.pano_cache_hot_order,
                &mut self.pano_cache_prefetch_order,
                &mut self.pano_cache,
                &mut self.pano_cache_bytes,
            );
        }
    }

    fn check_pano_timeouts(&mut self) {
        if let Some(job) = self.pano_inflight.clone() {
            if let Some(start) = self.pano_job_started.get(&job.room_id) {
                if start.elapsed() > Duration::from_secs(PANO_JOB_TIMEOUT_SECS) {
                    eprintln!(
                        "[pano] job timeout room={} gen={} elapsed_ms={} prefetch={}",
                        job.room_id,
                        job.generation,
                        start.elapsed().as_millis(),
                        job.prefetch
                    );
                    self.pano_job_started.remove(&job.room_id);
                    self.pano_inflight = None;
                    if let Some(entry) = self.pano_cache.get_mut(&job.room_id) {
                        entry.loading = false;
                    }
                    let attempts = self
                        .pano_job_attempts
                        .entry(job.room_id.clone())
                        .or_insert(0);
                    if *attempts < 1 {
                        *attempts += 1;
                        self.enqueue_pano_job(
                            &job.room_id,
                            &job.name,
                            job.source.clone(),
                            job.priority,
                            job.prefetch,
                        );
                    }
                    self.pano_generation = self.pano_generation.wrapping_add(1);
                }
            }
        }
        self.pano_jobs
            .retain(|j| j.generation == self.pano_generation);
    }

    fn pano_job_pending(&self, room_key: &str) -> bool {
        if self
            .pano_inflight
            .as_ref()
            .map(|r| r.room_id.eq_ignore_ascii_case(room_key))
            .unwrap_or(false)
        {
            return true;
        }
        self.pano_jobs
            .iter()
            .any(|j| j.room_id.eq_ignore_ascii_case(room_key))
    }

    fn enqueue_pano_job(
        &mut self,
        room_key: &str,
        name: &str,
        source: PanoSource,
        priority: PanoJobPriority,
        prefetch: bool,
    ) {
        let key = room_key.to_uppercase();
        if let Some(entry) = self.pano_cache.get_mut(&key) {
            if entry.frames.is_some() {
                return;
            }
            entry.loading = true;
        }
        if self
            .pano_inflight
            .as_ref()
            .map(|r| r.room_id.eq_ignore_ascii_case(&key))
            .unwrap_or(false)
        {
            return;
        }
        if let Some(existing) = self
            .pano_jobs
            .iter_mut()
            .find(|j| j.room_id.eq_ignore_ascii_case(&key))
        {
            if priority > existing.priority {
                existing.priority = priority;
            }
            existing.prefetch = existing.prefetch && prefetch;
            existing.generation = self.pano_generation;
            existing.name = name.to_string();
            existing.source = source;
            return;
        }
        if self.pano_jobs.len() >= PANO_QUEUE_LIMIT {
            if let Some((idx, worst)) = self
                .pano_jobs
                .iter()
                .enumerate()
                .min_by(|a, b| a.1.priority.cmp(&b.1.priority))
            {
                if worst.priority >= priority {
                    return;
                } else {
                    self.pano_jobs.remove(idx);
                }
            }
        }
        self.pano_jobs.push(PanoJob {
            room_id: key,
            name: name.to_string(),
            source,
            priority,
            generation: self.pano_generation,
            prefetch,
        });
    }

    fn kick_pano_worker(&mut self) {
        if self.pano_inflight.is_some() {
            return;
        }
        self.pano_jobs
            .retain(|j| j.generation == self.pano_generation);
        if self.pano_jobs.is_empty() {
            return;
        }
        let best_idx = self
            .pano_jobs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.priority.cmp(&b.1.priority).then_with(|| b.0.cmp(&a.0)))
            .map(|(i, _)| i)
            .unwrap_or(0);
        let job = self.pano_jobs.remove(best_idx);
        self.pano_job_started
            .insert(job.room_id.clone(), Instant::now());
        self.pano_inflight = Some(job.clone());
        let tx = self.pano_loader_tx.clone();
        let source_label = match &job.source {
            PanoSource::WebpBytes(_) => "webp",
            PanoSource::Mp4Path(_) => "mp4",
        };
        eprintln!(
            "[pano] job start room={} name={} gen={} prefetch={} source={}",
            job.room_id, job.name, job.generation, job.prefetch, source_label
        );
        thread::spawn(move || {
            let started = Instant::now();
            let frames = match job.source.clone() {
                PanoSource::WebpBytes(bytes) => {
                    load_webp_frames_from_bytes(bytes.as_slice()).map(|frames| AnimatedFrames {
                        name: job.name.clone(),
                        frames,
                    })
                }
                PanoSource::Mp4Path(path) => {
                    decode_mp4_frames(&path).map(|frames| AnimatedFrames {
                        name: job.name.clone(),
                        frames,
                    })
                }
            };
            let elapsed_ms = started.elapsed().as_millis();
            match &frames {
                Some(f) => eprintln!(
                    "[pano] job done room={} gen={} frames={} ms={} prefetch={}",
                    job.room_id,
                    job.generation,
                    f.frames.len(),
                    elapsed_ms,
                    job.prefetch
                ),
                None => eprintln!(
                    "[pano] job failed room={} gen={} ms={} prefetch={}",
                    job.room_id, job.generation, elapsed_ms, job.prefetch
                ),
            }
            let _ = tx.send(PanoLoadResult {
                room_id: job.room_id.clone(),
                name: job.name.clone(),
                frames: frames.map(Arc::new),
                generation: job.generation,
                prefetch: job.prefetch,
            });
        });
    }

    fn prefetch_room_pano(&mut self, room_key: &str) {
        let key = room_key.to_uppercase();
        let Some(manifest) = self.manifest.as_ref() else {
            return;
        };
        if self
            .pano_cache
            .get(&key)
            .map(|e| e.frames.is_some() || e.loading)
            .unwrap_or(false)
        {
            return;
        }
        if self.pano_job_pending(&key) {
            return;
        }
        let assets = resolve_room_assets(manifest, &key, &self.assets_base);
        let Some(pano_rel) = assets.pano else {
            return;
        };
        if let Some((first_frame, size, maybe_source, has_anim)) =
            load_pano_first_frame(&self.assets_base, &pano_rel)
        {
            let approx_bytes = estimate_frame_bytes(&first_frame);
            let entry = PanoCacheEntry {
                name: pano_rel.clone(),
                first_frame: first_frame.clone(),
                frames: None,
                size,
                loading: has_anim,
                prefetch: true,
                approx_bytes,
            };
            self.insert_pano_cache(key.clone(), entry);
            if has_anim {
                if let Some(source) = maybe_source {
                    self.enqueue_pano_job(&key, &pano_rel, source, PanoJobPriority::Neighbor, true);
                }
            }
        }
    }

    fn enqueue_neighbor_panos(&mut self) {
        let Some(parser) = self.parser.as_ref() else {
            return;
        };
        let Some(world) = parser.world.as_ref() else {
            return;
        };
        let Some(room_id) = self.ctx.room.as_ref() else {
            return;
        };
        let key = room_id.to_uppercase();
        let Some(room) = world.rooms.get(&key) else {
            return;
        };
        let mut targets: Vec<String> = room.exits.values().map(|t| t.to_uppercase()).collect();
        for eg in room.exits_guarded.iter() {
            if guard_true_ui(eg.guard.as_deref(), world, &self.engine) {
                targets.push(eg.to.to_uppercase());
            }
        }
        targets.retain(|t| t != &key);
        targets.sort();
        targets.dedup();
        for tgt in targets.into_iter().take(PANO_CACHE_PREFETCH_LIMIT) {
            self.prefetch_room_pano(&tgt);
        }
    }

    fn process_pano_loader(&mut self, ctx: &egui::Context) {
        loop {
            let msg = match self.pano_loader_rx.try_recv() {
                Ok(m) => m,
                Err(_) => break,
            };
            let started = self.pano_job_started.remove(&msg.room_id);
            self.pano_inflight = None;
            self.pano_job_attempts.remove(&msg.room_id);
            if msg.generation != self.pano_generation {
                eprintln!(
                    "[pano] drop stale result room={} msg_gen={} current_gen={} frames={}",
                    msg.room_id,
                    msg.generation,
                    self.pano_generation,
                    msg.frames.as_ref().map(|f| f.frames.len()).unwrap_or(0)
                );
                if let Some(entry) = self.pano_cache.get_mut(&msg.room_id) {
                    entry.loading = false;
                }
                continue;
            }
            let key = msg.room_id.clone();
            if let Some(frames) = msg.frames {
                if let Some(first) = frames.frames.first().cloned() {
                    let approx_bytes = estimate_frames_bytes(&frames);
                    let elapsed_ms = started.map(|s| s.elapsed().as_millis()).unwrap_or_default();
                    eprintln!(
                        "[pano] apply frames room={} gen={} frames={} prefetch={} elapsed_ms={}",
                        key,
                        msg.generation,
                        frames.frames.len(),
                        msg.prefetch,
                        elapsed_ms
                    );
                    let entry = PanoCacheEntry {
                        name: msg.name.clone(),
                        first_frame: first,
                        frames: Some(frames.clone()),
                        size: frames
                            .frames
                            .first()
                            .map(|f| f.image.size)
                            .unwrap_or([0, 0]),
                        loading: false,
                        prefetch: msg.prefetch,
                        approx_bytes,
                    };
                    self.insert_pano_cache(key.clone(), entry);
                    if let Some(current) = self.ctx.room.as_ref() {
                        if current.eq_ignore_ascii_case(&key) {
                            if let Some(entry_now) = self.pano_cache.get(&key).cloned() {
                                self.apply_pano_cache_entry(ctx, &key, &entry_now);
                                // Force the animation scheduler to kick immediately and advance once.
                                let now = ctx.input(|i| i.time);
                                self.pano_next_switch = (now - f64::EPSILON).max(0.0);
                                self.update_pano_animation(ctx);
                                ctx.request_repaint();
                            }
                        }
                    }
                }
            } else if let Some(entry) = self.pano_cache.get_mut(&key) {
                entry.loading = false;
                eprintln!(
                    "[pano] load failed room={} gen={} prefetch={} elapsed_ms={}",
                    key,
                    msg.generation,
                    msg.prefetch,
                    started.map(|s| s.elapsed().as_millis()).unwrap_or_default()
                );
            }
            self.pano_job_started.remove(&msg.room_id);
        }
    }

    fn render_map(&mut self, painter: &egui::Painter, rect: egui::Rect, response: &egui::Response) {
        let pal = self.current_palette();
        let bg = pal.map_bg;
        painter.rect_filled(rect, 4.0, bg);

        if response.dragged() {
            self.map_pan += response.drag_delta();
        }
        if response.hovered() {
            let scroll = response.ctx.input(|i| i.smooth_scroll_delta.y);
            if scroll.abs() > f32::EPSILON {
                let factor = (1.0 + scroll * 0.001).clamp(0.5, 1.5);
                self.map_zoom = (self.map_zoom * factor).clamp(0.3, 3.0);
            }
        }

        let Some(parser) = self.parser.as_ref() else {
            return;
        };
        let current_room = self.ctx.room.clone().unwrap_or_default();

        let norm_dir = |d: &str| -> String {
            match d.to_ascii_uppercase().as_str() {
                "N" | "NORTH" => "N".to_string(),
                "NE" | "NORTHEAST" => "NE".to_string(),
                "E" | "EAST" => "E".to_string(),
                "SE" | "SOUTHEAST" => "SE".to_string(),
                "S" | "SOUTH" => "S".to_string(),
                "SW" | "SOUTHWEST" => "SW".to_string(),
                "W" | "WEST" => "W".to_string(),
                "NW" | "NORTHWEST" => "NW".to_string(),
                "U" | "UP" => "UP".to_string(),
                "D" | "DOWN" => "DOWN".to_string(),
                "IN" | "ENTER" => "IN".to_string(),
                "OUT" | "EXIT" => "OUT".to_string(),
                other => other.to_string(),
            }
        };
        let dir_vec = |d: &str| -> egui::Vec2 {
            let deg: f32 = match norm_dir(d).as_str() {
                "N" => 0.0,
                "NE" => 45.0,
                "E" => 90.0,
                "SE" => 135.0,
                "S" => 180.0,
                "SW" => 225.0,
                "W" => 270.0,
                "NW" => 315.0,
                "UP" => -45.0,
                "DOWN" => 225.0,
                "IN" => 300.0,
                "OUT" => 60.0,
                _ => 25.0,
            };
            let rad = deg.to_radians();
            egui::Vec2::new(rad.sin(), -rad.cos())
        };

        // Collect seen nodes and visible edges
        let mut seen_nodes: HashSet<String> = HashSet::new();
        if !current_room.is_empty() {
            seen_nodes.insert(current_room.clone());
        }
        let visible_edges: Vec<MapEdgeState> = self
            .engine
            .map_state
            .edges
            .values()
            .filter(|e| e.visible)
            .cloned()
            .collect();
        for e in visible_edges.iter() {
            seen_nodes.insert(e.from.to_uppercase());
            if let Some(t) = e.to.as_ref() {
                seen_nodes.insert(t.to_uppercase());
            }
        }

        // For direct neighbors of the current room, capture all dirs to compute midpoint bearings.
        let mut neighbor_dirs: HashMap<String, Vec<String>> = HashMap::new();
        for e in visible_edges
            .iter()
            .filter(|e| e.from.eq_ignore_ascii_case(&current_room))
        {
            if let Some(to) = e.to.as_ref() {
                neighbor_dirs
                    .entry(to.to_uppercase())
                    .or_default()
                    .push(e.dir_raw.clone().unwrap_or(e.dir.clone()));
            }
        }

        // Ego-centric BFS over visible edges
        let mut depth: HashMap<String, usize> = HashMap::new();
        let mut parent: HashMap<String, String> = HashMap::new();
        let mut parent_dir: HashMap<String, String> = HashMap::new();
        let mut queue: VecDeque<String> = VecDeque::new();
        if !current_room.is_empty() {
            depth.insert(current_room.clone(), 0);
            queue.push_back(current_room.clone());
        }
        while let Some(r) = queue.pop_front() {
            let d = *depth.get(&r).unwrap_or(&0);
            for e in visible_edges
                .iter()
                .filter(|e| e.from.eq_ignore_ascii_case(&r))
            {
                if let Some(to) = e.to.as_ref() {
                    let to_u = to.to_uppercase();
                    if !depth.contains_key(&to_u) {
                        depth.insert(to_u.clone(), d + 1);
                        parent.insert(to_u.clone(), r.clone());
                        parent_dir.insert(
                            to_u.clone(),
                            e.dir_raw.as_deref().unwrap_or(&e.dir).to_string(),
                        );
                        queue.push_back(to_u);
                    }
                }
            }
        }

        // Position nodes based on path-composed bearing and depth
        let center = rect.center() + self.map_pan;
        let mut positions: HashMap<String, egui::Pos2> = HashMap::new();
        positions.insert(current_room.clone(), center);
        let base_radius = 50.0 * self.map_zoom;
        let step_radius = 80.0 * self.map_zoom;
        for id in seen_nodes.iter() {
            if id.eq_ignore_ascii_case(&current_room) {
                continue;
            }
            let d = depth.get(id).copied();
            let (angle_vec, radius) = if let Some(dep) = d {
                // If this is a direct neighbor of current, use midpoint of all dirs from current -> id.
                if dep == 1 {
                    if let Some(dirlist) = neighbor_dirs.get(id) {
                        let mut sum = egui::Vec2::ZERO;
                        for dlabel in dirlist {
                            sum += dir_vec(dlabel);
                        }
                        let final_vec = if sum.length_sq() < 0.0001 {
                            dirlist
                                .first()
                                .map(|d| dir_vec(d))
                                .unwrap_or_else(|| egui::Vec2::new(0.0, -1.0))
                        } else {
                            sum
                        };
                        let angle_unit = if final_vec.length_sq() > 0.0 {
                            final_vec / final_vec.length()
                        } else {
                            egui::Vec2::new(0.0, -1.0)
                        };
                        (angle_unit, base_radius + step_radius)
                    } else {
                        // fallback to path-based
                        let mut path_dirs: Vec<String> = Vec::new();
                        let mut cur = id.clone();
                        while let Some(p) = parent.get(&cur) {
                            if let Some(dir) = parent_dir.get(&cur) {
                                path_dirs.push(dir.clone());
                            }
                            cur = p.clone();
                        }
                        let mut sum = egui::Vec2::ZERO;
                        for dir in path_dirs.iter().rev() {
                            sum += dir_vec(dir);
                        }
                        let final_vec = if sum.length_sq() < 0.0001 {
                            path_dirs
                                .last()
                                .map(|d| dir_vec(d))
                                .unwrap_or_else(|| egui::Vec2::new(0.0, -1.0))
                        } else {
                            sum
                        };
                        let angle_unit = if final_vec.length_sq() > 0.0 {
                            final_vec / final_vec.length()
                        } else {
                            egui::Vec2::new(0.0, -1.0)
                        };
                        (angle_unit, base_radius + step_radius * dep as f32)
                    }
                } else {
                    let mut path_dirs: Vec<String> = Vec::new();
                    let mut cur = id.clone();
                    while let Some(p) = parent.get(&cur) {
                        if let Some(dir) = parent_dir.get(&cur) {
                            path_dirs.push(dir.clone());
                        }
                        cur = p.clone();
                    }
                    let mut sum = egui::Vec2::ZERO;
                    for dir in path_dirs.iter().rev() {
                        sum += dir_vec(dir);
                    }
                    let final_vec = if sum.length_sq() < 0.0001 {
                        path_dirs
                            .last()
                            .map(|d| dir_vec(d))
                            .unwrap_or_else(|| egui::Vec2::new(0.0, -1.0))
                    } else {
                        sum
                    };
                    let angle_unit = if final_vec.length_sq() > 0.0 {
                        final_vec / final_vec.length()
                    } else {
                        egui::Vec2::new(0.0, -1.0)
                    };
                    (angle_unit, base_radius + step_radius * dep as f32)
                }
            } else {
                use std::collections::hash_map::DefaultHasher;
                use std::hash::{Hash, Hasher};
                let mut hasher = DefaultHasher::new();
                id.hash(&mut hasher);
                let hash = hasher.finish();
                let ang = (hash as f32 / u64::MAX as f32) * std::f32::consts::TAU;
                (
                    egui::Vec2::new(ang.sin(), -ang.cos()),
                    base_radius + step_radius * 5.0,
                )
            };
            positions.insert(id.clone(), center + angle_vec * radius);
        }

        // Draw nodes (background layer for circles)
        for id in seen_nodes.iter() {
            let pos = positions.get(id).cloned().unwrap_or(center);
            let radius = 8.0 * self.map_zoom;
            let is_current = self
                .ctx
                .room
                .as_ref()
                .map(|r| r.eq_ignore_ascii_case(id))
                .unwrap_or(false);
            let color = if is_current {
                pal.map_node_current
            } else {
                pal.map_node_seen
            };
            painter.circle_filled(pos, radius, color);
        }

        // Fan edges with slight offset per (from, dir) and curve toward the target.
        let mut edge_counts: HashMap<(String, String), usize> = HashMap::new();
        for e in visible_edges.iter() {
            let key = (
                e.from.to_uppercase(),
                norm_dir(e.dir_raw.as_deref().unwrap_or(&e.dir)),
            );
            *edge_counts.entry(key).or_insert(0) += 1;
        }
        let mut edge_used: HashMap<(String, String), usize> = HashMap::new();
        let mut arrow_heads: Vec<(egui::Pos2, egui::Pos2, egui::Pos2)> = Vec::new();

        for e in visible_edges.iter() {
            let from_u = e.from.to_uppercase();
            let dir_label = e.dir_raw.as_deref().unwrap_or(&e.dir);
            let canon = norm_dir(dir_label);
            let key = (from_u.clone(), canon.clone());
            let idx = edge_used
                .entry(key.clone())
                .and_modify(|v| *v += 1)
                .or_insert(0);
            let total = *edge_counts.get(&key).unwrap_or(&1);
            let offset_idx = *idx as isize - (total as isize - 1) / 2;

            let from_pos = *positions.get(&from_u).unwrap_or(&center);
            let to_pos =
                e.to.as_ref()
                    .and_then(|t| positions.get(&t.to_uppercase()).cloned())
                    .unwrap_or_else(|| from_pos + dir_vec(dir_label) * (base_radius + step_radius));

            let dir_unit = {
                let v = dir_vec(dir_label);
                if v.length_sq() > 0.0 {
                    v / v.length()
                } else {
                    egui::Vec2::new(0.0, -1.0)
                }
            };
            let perp = egui::Vec2::new(-dir_unit.y, dir_unit.x);
            let fan_offset = perp * (offset_idx as f32 * 14.0 * self.map_zoom);

            let start = from_pos + fan_offset;
            let end = to_pos + fan_offset;
            let chord = end - start;
            let ctrl = start + dir_unit * (chord.length() * 0.35) + fan_offset * 0.2;

            let stroke = egui::Stroke::new(1.5, pal.map_edge);
            painter.line_segment([start, ctrl], stroke);
            painter.line_segment([ctrl, end], stroke);

            let seg = end - ctrl;
            if seg.length_sq() > 1.0 {
                let dir_n = seg / seg.length();
                let tip = end;
                let back = dir_n * (10.0 * self.map_zoom);
                let pp = egui::Vec2::new(-dir_n.y, dir_n.x) * (6.0 * self.map_zoom);
                arrow_heads.push((tip, tip - back + pp, tip - back - pp));
            }

            // Quadratic bezier midpoint at t=0.5
            // Label anchored near the control point (unique per direction), offset perpendicular to the initial direction.
            let label_base = ctrl;
            let label_perp = egui::Vec2::new(-dir_unit.y, dir_unit.x);
            let label_offset = label_perp * (12.0 * self.map_zoom);
            painter.text(
                label_base + label_offset,
                egui::Align2::CENTER_CENTER,
                canon,
                egui::FontId::proportional(12.0 * self.map_zoom),
                pal.fg,
            );
        }

        // Arrow heads on foreground layer with contrast color.
        let head_stroke = egui::Stroke::new(1.5, pal.fg_contrast);
        for (tip, a, b) in arrow_heads {
            painter.line_segment([tip, a], head_stroke);
            painter.line_segment([tip, b], head_stroke);
        }

        // Labels (foreground)
        for id in seen_nodes.iter() {
            let pos = positions.get(id).cloned().unwrap_or(center);
            let radius = 8.0 * self.map_zoom;
            let is_current = self
                .ctx
                .room
                .as_ref()
                .map(|r| r.eq_ignore_ascii_case(id))
                .unwrap_or(false);
            if !is_current {
                let name_owned = self
                    .engine
                    .map_state
                    .nodes
                    .get(id)
                    .and_then(|n| n.name.clone())
                    .or_else(|| {
                        parser
                            .world
                            .as_ref()
                            .and_then(|w| w.rooms.get(id).and_then(|r| r.desc.clone()))
                    })
                    .unwrap_or_else(|| id.to_string());
                painter.text(
                    pos + egui::vec2(0.0, radius + 4.0),
                    egui::Align2::CENTER_TOP,
                    name_owned,
                    egui::FontId::proportional(12.0 * self.map_zoom),
                    pal.fg,
                );
            }
        }
    }

    fn font_label(&self, selection: Option<&str>) -> String {
        if let Some(path) = selection {
            Path::new(path)
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or(path)
                .to_string()
        } else {
            "Default".into()
        }
    }

    fn text_formats(&self) -> (TextFormat, TextFormat) {
        let pal = self.current_palette();
        let base_family = if self.font_error.is_none() && self.font_regular.is_some() {
            FontFamily::Name(FONT_FAMILY_BASE.into())
        } else {
            FontFamily::Proportional
        };
        let bold_family = if self.font_error.is_none() && self.font_bold.is_some() {
            FontFamily::Name(FONT_FAMILY_BOLD.into())
        } else {
            base_family.clone()
        };
        let base_font = FontId::new(self.font_size_base, base_family.clone());
        let emphasis_font = FontId::new(self.font_size_bold, bold_family);
        let normal = TextFormat {
            color: pal.fg,
            font_id: base_font,
            ..Default::default()
        };
        let emphasis = TextFormat {
            color: pal.fg_contrast,
            font_id: emphasis_font,
            line_height: Some(self.font_size_bold + 3.0),
            ..Default::default()
        };
        (normal, emphasis)
    }

    fn current_palette(&self) -> ThemePalette {
        self.palette.data()
    }

    fn apply_palette_visuals(&self, ctx: &egui::Context) {
        let pal = self.current_palette();
        let mut visuals = Visuals::dark();
        visuals.override_text_color = None; // allow per-text colors (narrator contrast, highlights)
        visuals.widgets.inactive.bg_fill = pal.widget_inactive;
        visuals.widgets.hovered.bg_fill = pal.widget_hover;
        visuals.widgets.active.bg_fill = pal.widget_active;
        visuals.panel_fill = pal.panel;
        visuals.widgets.noninteractive.fg_stroke.color = pal.fg;
        visuals.widgets.inactive.fg_stroke.color = pal.fg;
        visuals.widgets.hovered.fg_stroke.color = pal.fg;
        visuals.widgets.active.fg_stroke.color = pal.fg;
        ctx.set_visuals(visuals);
    }

    fn apply_fonts(&mut self, ctx: &egui::Context) {
        let mut defs = FontDefinitions::default();
        defs.font_data.clear();
        defs.families.insert(FontFamily::Proportional, vec![]);
        defs.families.insert(FontFamily::Monospace, vec![]);
        defs.families
            .insert(FontFamily::Name(FONT_FAMILY_BASE.into()), vec![]);
        defs.families
            .insert(FontFamily::Name(FONT_FAMILY_BOLD.into()), vec![]);
        defs.families
            .insert(FontFamily::Name(FONT_FAMILY_BOLD_UI.into()), vec![]);

        let mut attached = false;

        if let Some(base_path) = self.font_regular.as_ref() {
            if let Some(data) = load_font_data(base_path) {
                defs.font_data.insert(FONT_FAMILY_BASE.into(), data);
                defs.families
                    .entry(FontFamily::Proportional)
                    .or_default()
                    .insert(0, FONT_FAMILY_BASE.into());
                defs.families
                    .entry(FontFamily::Monospace)
                    .or_default()
                    .insert(0, FONT_FAMILY_BASE.into());
                defs.families
                    .entry(FontFamily::Name(FONT_FAMILY_BASE.into()))
                    .or_default()
                    .insert(0, FONT_FAMILY_BASE.into());
                attached = true;
            } else {
                self.font_error = Some(format!("Failed to load font: {}", base_path));
            }
        }

        let bold_path = self
            .font_bold
            .as_ref()
            .or_else(|| self.font_regular.as_ref());
        if let Some(bold) = bold_path {
            if let Some(data) = load_font_data(bold) {
                let yoff = -(BOLD_Y_OFFSET_PX / self.font_size_bold.max(1.0));
                let data = data.tweak(FontTweak {
                    y_offset_factor: yoff,
                    baseline_offset_factor: yoff,
                    ..FontTweak::default()
                });
                defs.font_data.insert(FONT_FAMILY_BOLD.into(), data);
                defs.families
                    .entry(FontFamily::Proportional)
                    .or_default()
                    .insert(0, FONT_FAMILY_BOLD.into());
                defs.families
                    .entry(FontFamily::Monospace)
                    .or_default()
                    .insert(0, FONT_FAMILY_BOLD.into());
                defs.families
                    .entry(FontFamily::Name(FONT_FAMILY_BOLD.into()))
                    .or_default()
                    .insert(0, FONT_FAMILY_BOLD.into());
                // UI bold: same font, no vertical tweak so buttons/headings stay centered.
                if let Some(ui_data) = load_font_data(bold) {
                    defs.font_data.insert(FONT_FAMILY_BOLD_UI.into(), ui_data);
                    defs.families
                        .entry(FontFamily::Proportional)
                        .or_default()
                        .insert(0, FONT_FAMILY_BOLD_UI.into());
                    defs.families
                        .entry(FontFamily::Monospace)
                        .or_default()
                        .insert(0, FONT_FAMILY_BOLD_UI.into());
                    defs.families
                        .entry(FontFamily::Name(FONT_FAMILY_BOLD_UI.into()))
                        .or_default()
                        .insert(0, FONT_FAMILY_BOLD_UI.into());
                }
                attached = true;
            } else {
                self.font_error = Some(format!("Failed to load font: {}", bold));
            }
        }

        if attached {
            self.font_error = None;
            ctx.set_fonts(defs);
        } else {
            self.font_error =
                Some("Using Rusty Knife default fonts (no custom fonts loaded)".into());
            ctx.set_fonts(FontDefinitions::default());
        }

        // Nudge headings/buttons to use the untweaked bold family so they stay vertically centered.
        let mut style = (*ctx.style()).clone();
        if let Some(h) = style.text_styles.get_mut(&TextStyle::Heading) {
            h.family = FontFamily::Name(FONT_FAMILY_BOLD_UI.into());
        }
        if let Some(b) = style.text_styles.get_mut(&TextStyle::Button) {
            b.family = FontFamily::Name(FONT_FAMILY_BOLD_UI.into());
        }
        ctx.set_style(style);
    }

    fn push_plain(&mut self, s: impl Into<String>) {
        self.log.insert(0, LogEntry::Plain(s.into()));
    }

    fn push_styled(&mut self, plain: String, job: LayoutJob) {
        self.log.insert(0, LogEntry::Styled { job, plain });
    }

    fn persist_settings(&self) {
        let settings = SavedSettings {
            tts_mode: self.tts_mode,
            tts_volume: self.tts_volume,
            palette: self.palette,
            scene_mode: self.scene_mode,
            scene_drag_sensitivity: self.scene_drag_sensitivity,
            scene_invert_y: self.scene_invert_y,
            scene_fov_y_deg: self.scene_fov_y_deg,
            show_hints: self.show_hints,
            show_system_messages: self.show_system_messages,
            font_regular: self.font_regular.clone(),
            font_bold: self.font_bold.clone(),
            font_size_base: self.font_size_base,
            font_size_bold: self.font_size_bold,
            save_path: self.save_path.clone(),
            speech_enabled: self.speech_enabled,
            speech_auto_send: self.speech_auto_send,
            speech_model_path: self.speech_model_path.clone(),
            speech_language: self.speech_language.clone(),
            speech_hotkey: self.speech_hotkey.clone(),
        };
        save_settings(&self.settings_path, &settings);
    }

    fn available_exits(&self) -> HashSet<String> {
        if let (Some(parser), Some(room_id)) = (&self.parser, &self.ctx.room) {
            if let Some(world) = parser.world.as_ref() {
                if let Some(room) = world.rooms.get(&room_id.to_uppercase()) {
                    let mut set = HashSet::new();
                    for d in room.exits.keys() {
                        set.insert(d.to_uppercase());
                    }
                    for eg in room.exits_guarded.iter() {
                        if let Some(g) = eg.guard.as_deref() {
                            if g.eq_ignore_ascii_case("UP-CHIMNEY-FUNCTION") {
                                let inv_len = self.ctx.inventory.len();
                                let has_lamp = self
                                    .ctx
                                    .inventory
                                    .iter()
                                    .any(|i| i.eq_ignore_ascii_case("LAMP"));
                                // ZIL allows up only if carrying the lamp and at most two items, and not empty-handed.
                                if inv_len == 0 || inv_len > 2 || !has_lamp {
                                    continue;
                                }
                            }
                        }
                        if guard_true_ui(eg.guard.as_deref(), world, &self.engine) {
                            set.insert(eg.dir.to_uppercase());
                        }
                    }
                    return set;
                }
            }
            if let Some(idx) = parser.world_index.as_ref() {
                if let Some(set) = idx.room_exits.get(&room_id.to_uppercase()) {
                    return set.clone();
                }
            }
        }
        HashSet::new()
    }

    fn try_save(&mut self, path: &str) {
        if let Some(parser) = self.parser.as_ref() {
            if let Some(world) = parser.world.as_ref() {
                match save_to_file_core(path, &self.ctx, world, &self.engine) {
                    Ok(_) => self.push_plain(format!("(options) Saved to {}", path)),
                    Err(e) => self.push_plain(format!("(options) Save failed: {}", e)),
                }
            } else {
                self.push_plain("(options) World not loaded; cannot save");
            }
        } else {
            self.push_plain("(options) Parser not loaded; cannot save");
        }
    }

    fn try_load(&mut self, path: &str) {
        if let Some(parser) = self.parser.as_mut() {
            if let Some(world) = parser.world.as_mut() {
                match load_save_from_file(path, &mut self.ctx, world, Some(&mut self.engine)) {
                    Ok(save) => {
                        self.refresh_world_views();
                        self.engine.tick = save.ticks;
                        self.push_plain(format!(
                            "(options) Loaded save game={:?} room={:?} inv={}",
                            save.game,
                            save.room,
                            save.inventory.len()
                        ));
                        self.record_status(StatusLine {
                            message: "(load) applied save".into(),
                            success: true,
                        });
                        self.log_room_description(true, false, true);
                    }
                    Err(e) => {
                        self.push_plain(format!("(options) Load failed: {}", e));
                        self.record_status(StatusLine {
                            message: "(load) failed".into(),
                            success: false,
                        });
                    }
                }
            } else {
                self.push_plain("(options) World not loaded; cannot load");
            }
        } else {
            self.push_plain("(options) Parser not loaded; cannot load");
        }
    }

    fn has_exit(&self, exits: &HashSet<String>, aliases: &[&str]) -> bool {
        aliases.iter().any(|a| exits.contains(&a.to_uppercase()))
    }

    fn send_command(&mut self, cmd: &str, ctx: &egui::Context) {
        self.input = cmd.to_string();
        self.handle_input(ctx);
    }

    fn is_move_token(tok: &str) -> bool {
        matches!(
            tok.to_uppercase().as_str(),
            "GO" | "WALK"
                | "RUN"
                | "ENTER"
                | "EXIT"
                | "LEAVE"
                | "CLIMB"
                | "N"
                | "S"
                | "E"
                | "W"
                | "NE"
                | "NW"
                | "SE"
                | "SW"
                | "NORTH"
                | "SOUTH"
                | "EAST"
                | "WEST"
                | "NORTHEAST"
                | "NORTHWEST"
                | "SOUTHEAST"
                | "SOUTHWEST"
                | "U"
                | "UP"
                | "D"
                | "DOWN"
                | "IN"
                | "OUT"
        )
    }

    fn maybe_apply_move(
        &mut self,
        intent: &ScoredIntent,
        ctx: &egui::Context,
    ) -> Option<(StatusLine, bool)> {
        let parser = self.parser.as_ref()?;
        let world = parser.world.as_ref()?;
        if let Some(n) = intent.noun.as_ref() {
            if let Some(obj_id) = self.resolve_object_id(&n.token) {
                if let Some(obj) = world.objects.get(&obj_id) {
                    let intent_verb = intent
                        .verb
                        .as_ref()
                        .map(|v| v.token.to_uppercase())
                        .unwrap_or_default();
                    let verbs: Vec<String> = obj.verbs.iter().map(|v| v.to_uppercase()).collect();
                    if !intent_verb.is_empty()
                        && !verbs.is_empty()
                        && verbs.iter().any(|v| v == &intent_verb)
                    {
                        return None;
                    }
                }
            }
        }
        let mut dir: Option<String> = intent.noun.as_ref().map(|c| c.token.clone());
        if dir.is_none() {
            if let Some(v) = intent.verb.as_ref() {
                if Self::is_move_token(&v.token) {
                    dir = Some(v.token.clone());
                }
            }
        }
        let dir = dir?;
        match move_player(&mut self.ctx, world, Some(&mut self.engine), &dir) {
            Ok(dest) => {
                self.reload_room_media(ctx);
                self.current_dialog = None;
                self.engine.dialog = None;
                self.last_move_dir = Some(self.prettify_exit(&dir));
                Some((
                    StatusLine {
                        message: format!(
                            "(move {}) moved to {}",
                            dir.to_lowercase(),
                            self.prettify_id(&dest)
                        ),
                        success: true,
                    },
                    true,
                ))
            }
            Err(e) => Some((
                StatusLine {
                    message: format!("(move) {}", e),
                    success: false,
                },
                false,
            )),
        }
    }

    fn maybe_apply_move_tokens(
        &mut self,
        tokens: &[&str],
        ctx: &egui::Context,
    ) -> Option<(StatusLine, bool)> {
        if tokens.len() != 1 {
            return None;
        }
        let t = tokens[0].to_uppercase();
        if !matches!(
            t.as_str(),
            "IN" | "OUT"
                | "UP"
                | "DOWN"
                | "U"
                | "D"
                | "N"
                | "S"
                | "E"
                | "W"
                | "NE"
                | "NW"
                | "SE"
                | "SW"
        ) {
            return None;
        }
        let parser = self.parser.as_ref()?;
        let world = parser.world.as_ref()?;
        match move_player(&mut self.ctx, world, Some(&mut self.engine), &t) {
            Ok(dest) => {
                self.reload_room_media(ctx);
                self.current_dialog = None;
                self.engine.dialog = None;
                self.last_move_dir = Some(self.prettify_exit(&t));
                Some((
                    StatusLine {
                        message: format!(
                            "(move {}) moved to {}",
                            t.to_lowercase(),
                            self.prettify_id(&dest)
                        ),
                        success: true,
                    },
                    true,
                ))
            }
            Err(e) => Some((
                StatusLine {
                    message: format!("(move) {}", e),
                    success: false,
                },
                false,
            )),
        }
    }

    fn handle_hotkeys(&mut self, ctx: &egui::Context) {
        if self.input_has_focus {
            return;
        }
        let mut pressed: Option<&'static str> = None;
        ctx.input(|i| {
            let map = [
                (egui::Key::Num7, "nw"),
                (egui::Key::Num8, "north"),
                (egui::Key::Num9, "ne"),
                (egui::Key::Num4, "west"),
                (egui::Key::Num5, "look"),
                (egui::Key::Num6, "east"),
                (egui::Key::Num1, "sw"),
                (egui::Key::Num2, "south"),
                (egui::Key::Num3, "se"),
                (egui::Key::Plus, "up"),
                (egui::Key::Minus, "down"),
                (egui::Key::Slash, "in"),
                (egui::Key::Backslash, "out"),
                (egui::Key::ArrowUp, "up"),
                (egui::Key::PageUp, "up"),
                (egui::Key::ArrowDown, "down"),
                (egui::Key::PageDown, "down"),
                (egui::Key::U, "up"),
                (egui::Key::D, "down"),
                (egui::Key::I, "in"),
                (egui::Key::O, "out"),
            ];
            for (key, cmd) in map {
                if i.key_pressed(key) {
                    pressed = Some(cmd);
                    break;
                }
            }
        });

        // Text-event fallback to capture '+'/'-' '/' '*' when not mapped as keys
        ctx.input(|i| {
            for ev in &i.events {
                if let egui::Event::Text(t) = ev {
                    if t == "+" {
                        pressed = Some("up");
                        break;
                    }
                    if t == "-" {
                        pressed = Some("down");
                        break;
                    }
                    if t == "/" {
                        pressed = Some("in");
                        break;
                    }
                    if t == "*" {
                        pressed = Some("out");
                        break;
                    }
                }
            }
        });
        if let Some(cmd) = pressed {
            self.send_command(cmd, ctx);
        }
    }

    fn speech_hotkey_active(&self, ctx: &egui::Context) -> bool {
        let key = self.speech_hotkey.to_ascii_lowercase();
        ctx.input(|i| match key.as_str() {
            "leftctrl" | "ctrl" | "control" => i.modifiers.ctrl,
            "space" => i.key_down(egui::Key::Space),
            _ => false,
        })
    }

    fn handle_speech_hotkey(&mut self, ctx: &egui::Context) {
        if !self.speech_enabled || !SPEECH_BUILD_ENABLED {
            self.speech_hotkey_down = false;
            return;
        }
        let down = self.speech_hotkey_active(ctx);
        if down && !self.speech_hotkey_down {
            self.start_speech_capture();
            self.speech_hotkey_down = true;
        } else if !down && self.speech_hotkey_down {
            self.stop_speech_capture();
            self.speech_hotkey_down = false;
        }
    }

    fn handle_input(&mut self, ctx: &egui::Context) {
        if self.input.trim().is_empty() {
            return;
        }
        self.engine.tick = self.engine.tick.wrapping_add(1);
        let line = std::mem::take(&mut self.input);
        self.push_plain(format!("> {}", line));

        if line.starts_with('/') || line.starts_with(':') {
            self.handle_meta(&line[1..], ctx);
            return;
        }

        if let Some(parser) = self.parser.as_ref() {
            if let Some(world) = parser.world.as_ref() {
                if let Some(result) =
                    apply_builder_action(world, &self.ctx, &mut self.engine, line.trim())
                {
                    match result {
                        Ok(msg) => self.push_plain(self.prettify_sentence(&msg)),
                        Err(msg) => self.push_plain(self.prettify_sentence(&msg)),
                    }
                    return;
                }
            }
        }

        if let Some(parser) = self.parser.as_ref() {
            let normalized = normalize_player_input(&line, parser);
            let tokens: Vec<&str> = normalized.tokens.iter().map(|s| s.as_str()).collect();
            let mut move_success = false;
            let status = if let Some(intent) = match_intent(&tokens, parser, Some(&self.ctx)) {
                if self.show_system_messages {
                    self.push_plain(format!(
                        "Intent: verb={:?} noun={:?} prep={:?} io={:?} conf={:.2}",
                        intent.verb.as_ref().map(|c| &c.token),
                        intent.noun.as_ref().map(|c| &c.token),
                        intent.prep,
                        intent.indirect.as_ref().map(|c| &c.token),
                        intent.confidence
                    ));
                }
                let mut handled = false;
                if self.handle_look_intent(&intent) {
                    handled = true;
                }
                if !handled {
                    if self.handle_action_intent(&intent) {
                        handled = true;
                    }
                }
                if !handled {
                    if let Some((st, ok)) = self.maybe_apply_move(&intent, ctx) {
                        move_success = ok;
                        Some(st)
                    } else {
                        None
                    }
                } else {
                    None
                }
            } else {
                // fallback: try pure direction tokens (including IN/OUT) even when intent not resolved
                let status_move = self.maybe_apply_move_tokens(&tokens, ctx);
                if status_move.is_none() && self.show_system_messages {
                    self.push_plain("Intent: (no guess)");
                }
                if let Some((_, ok)) = status_move.as_ref() {
                    move_success = *ok;
                }
                status_move.map(|(st, _)| st)
            };
            if !normalized.corrections.is_empty() && self.show_system_messages {
                self.push_plain(format!(
                    "Corrections: {}",
                    normalized.corrections.join("; ")
                ));
            }
            if let Some(s) = status {
                self.record_status(s);
                if move_success {
                    self.log_room_description(true, false, true);
                }
            }
        } else {
            self.push_plain("(parser not loaded)");
        }
        self.handle_world_events(ctx);
    }

    fn reload_room_media(&mut self, ctx: &egui::Context) {
        let Some(manifest) = self.manifest.as_ref() else {
            return;
        };
        let room_id = match self.ctx.room.as_ref() {
            Some(r) => r.clone(),
            None => return,
        };
        self.pano_generation = self.pano_generation.wrapping_add(1);
        self.pano_jobs.clear();
        self.pano_inflight = None;
        self.pano_job_started.clear();
        self.pano_frames = None;
        self.pano_texture = None;
        self.pano_size = None;
        self.pano_rel = None;
        self.pano_frame_index = 0;
        self.pano_next_switch = 0.0;
        let room_key = room_id.to_uppercase();
        let assets = resolve_room_assets(manifest, &room_key, &self.assets_base);

        self.hero_texture = assets
            .hero
            .clone()
            .or(assets.thumb.clone())
            .and_then(|rel| load_static_image(&self.assets_base, &rel))
            .map(|(name, img)| ctx.load_texture(name, img, egui::TextureOptions::LINEAR));

        if let Some(pano_rel) = assets.pano.clone() {
            self.pano_rel = Some(pano_rel.clone());
            if let Some(entry) = self.pano_cache.get(&room_key).cloned() {
                self.apply_pano_cache_entry(ctx, &room_key, &entry);
                let job_active = self.pano_job_pending(&room_key);
                if entry.frames.is_some() {
                    let now = ctx.input(|i| i.time);
                    self.pano_next_switch = (now - f64::EPSILON).max(0.0);
                    self.update_pano_animation(ctx);
                    ctx.request_repaint();
                    return;
                }
                if entry.loading && job_active {
                    return;
                }
                if let Some(e) = self.pano_cache.get_mut(&room_key) {
                    e.loading = false;
                }
            }
            if let Some((first_frame, size, maybe_bytes, has_anim)) =
                load_pano_first_frame(&self.assets_base, &pano_rel)
            {
                let approx_bytes = estimate_frame_bytes(&first_frame);
                let entry = PanoCacheEntry {
                    name: pano_rel.clone(),
                    first_frame: first_frame.clone(),
                    frames: None,
                    size,
                    loading: has_anim,
                    prefetch: false,
                    approx_bytes,
                };
                self.insert_pano_cache(room_key.clone(), entry);
                if let Some(entry_now) = self.pano_cache.get(&room_key).cloned() {
                    self.apply_pano_cache_entry(ctx, &room_key, &entry_now);
                }
                if has_anim {
                    if let Some(source) = maybe_bytes {
                        self.enqueue_pano_job(
                            &room_key,
                            &pano_rel,
                            source,
                            PanoJobPriority::Current,
                            false,
                        );
                    }
                }
            }
        }
        self.enqueue_neighbor_panos();
    }
    fn handle_meta(&mut self, cmdline: &str, ctx: &egui::Context) {
        let mut parts = cmdline.trim().splitn(2, char::is_whitespace);
        let cmd = parts.next().unwrap_or("").to_lowercase();
        let rest = parts.next().unwrap_or("").trim();
        match cmd.as_str() {
            "room" => {
                if rest.is_empty() {
                    self.push_plain("Usage: /room ROOM_ID");
                    return;
                }
                let room_id = normalize_room_id(rest);
                if let Some(p) = self.parser.as_ref() {
                    let ok = p
                        .world
                        .as_ref()
                        .map(|w| w.rooms.contains_key(&room_id))
                        .unwrap_or(true);
                    if ok {
                        self.ctx.room = Some(room_id.clone());
                        self.push_plain(format!("(context) room set to {}", room_id));
                        self.reload_room_media(ctx);
                        self.log_room_description(true, false, false);
                    } else {
                        self.push_plain(format!("(context) room '{}' not found", room_id));
                    }
                }
            }
            "inv" => {
                if rest.is_empty() {
                    self.push_plain("Usage: /inv +OBJECT_ID | -OBJECT_ID | /inv clear");
                    return;
                }
                if rest.eq_ignore_ascii_case("clear") {
                    self.ctx.inventory.clear();
                    self.push_plain("(context) inventory cleared");
                    return;
                }
                let action = rest.chars().next().unwrap_or(' ');
                let payload = rest[1..].trim();
                if payload.is_empty() {
                    self.push_plain("Usage: /inv +OBJECT_ID | -OBJECT_ID | /inv clear");
                    return;
                }
                match action {
                    '+' => {
                        self.ctx.inventory.push(payload.to_uppercase());
                        self.ctx.inventory.sort();
                        self.ctx.inventory.dedup();
                        self.push_plain(format!("(context) added to inventory: {}", payload));
                    }
                    '-' => {
                        let before = self.ctx.inventory.len();
                        self.ctx
                            .inventory
                            .retain(|o| o != payload && o != &payload.to_uppercase());
                        if self.ctx.inventory.len() < before {
                            self.push_plain(format!(
                                "(context) removed from inventory: {}",
                                payload
                            ));
                        } else {
                            self.push_plain(format!(
                                "(context) not found in inventory: {}",
                                payload
                            ));
                        }
                    }
                    _ => {
                        self.push_plain("Usage: /inv +OBJECT_ID | -OBJECT_ID | /inv clear");
                    }
                }
            }
            "newworld" => {
                if self.parser.is_none() {
                    self.push_plain("(builder) parser not loaded");
                    return;
                }
                let name_raw = if rest.is_empty() { "builder" } else { rest };
                let (world_path, log_path, label) = {
                    let candidate = PathBuf::from(name_raw);
                    if candidate.extension().is_some() {
                        let log = candidate.with_extension("log.jsonl");
                        let label = candidate
                            .file_stem()
                            .and_then(|s| s.to_str())
                            .unwrap_or("builder")
                            .to_string();
                        (candidate, log, label)
                    } else {
                        let base = normalize_id(name_raw).to_lowercase();
                        let dir = PathBuf::from("../out/worlds");
                        let _ = fs::create_dir_all(&dir);
                        (
                            dir.join(format!("{}.json", base)),
                            dir.join(format!("{}.log.jsonl", base)),
                            base,
                        )
                    }
                };
                if let Some(dir) = world_path.parent() {
                    let _ = fs::create_dir_all(dir);
                }
                if let Some(dir) = log_path.parent() {
                    let _ = fs::create_dir_all(dir);
                }
                self.builder_world_path = Some(world_path.clone());
                self.builder_log_path = Some(log_path.clone());
                let mut world = blank_world(&label);
                if let Some(meta) = world.builder_meta.as_mut() {
                    meta.log_path = Some(log_path.to_string_lossy().to_string());
                    meta.name.get_or_insert(label.clone());
                }
                let change = BuilderChange {
                    ts: None,
                    tick: self.engine.tick,
                    user_cmd: cmdline.to_string(),
                    effect: "Initialized new world".into(),
                    room: None,
                };
                world.builder_log.push(change.clone());
                if let Some(parser) = self.parser.as_mut() {
                    parser.world = Some(world);
                }
                self.refresh_world_views();
                self.ctx = ParseContext {
                    room: Some(normalize_id("START")),
                    inventory: vec![],
                };
                self.engine = init_engine(EngineConfig {
                    data_path: None,
                    media_enabled: true,
                });
                self.current_dialog = None;
                self.persist_builder_change(Some(&change));
                self.push_plain(format!(
                    "(builder) started new world '{}' -> {} (log {})",
                    label,
                    world_path.display(),
                    log_path.display()
                ));
                self.engine.dialog = None;
                self.reload_room_media(ctx);
                self.log_room_description(true, false, false);
            }
            "carve" => {
                if rest.is_empty() {
                    self.push_plain("Usage: /carve <direction>");
                    return;
                }
                let room_id = if let Some(r) = self.ctx.room.as_ref() {
                    r.to_string()
                } else {
                    self.push_plain("(builder) no current room; set one with /room");
                    return;
                };
                let Some(parser) = self.parser.as_mut() else {
                    self.push_plain("(builder) parser not loaded");
                    return;
                };
                let Some(world) = parser.world.as_mut() else {
                    self.push_plain("(builder) no world loaded; use /newworld first");
                    return;
                };
                match carve_exit(world, &room_id, rest, None, true, self.engine.tick, cmdline) {
                    Ok(change) => {
                        self.refresh_world_views();
                        self.persist_builder_change(Some(&change));
                        self.push_plain(format!("(builder) {}", change.effect));
                    }
                    Err(e) => self.push_plain(format!("(builder) {}", e)),
                }
            }
            "exit" => {
                if rest.is_empty() {
                    self.push_plain("Usage: /exit <dir> <dest> [guard_flag] [message] [oneway]");
                    return;
                }
                let mut tokens: Vec<&str> = rest.split_whitespace().collect();
                let mut oneway = false;
                if tokens
                    .last()
                    .map(|t| t.eq_ignore_ascii_case("oneway"))
                    .unwrap_or(false)
                {
                    tokens.pop();
                    oneway = true;
                }
                if tokens.len() < 2 {
                    self.push_plain("Usage: /exit <dir> <dest> [guard_flag] [message] [oneway]");
                    return;
                }
                let dir = tokens[0];
                let dest = tokens[1];
                let guard_flag = if tokens.len() > 2 {
                    let g = tokens[2].trim();
                    if g.eq_ignore_ascii_case("none") || g.is_empty() {
                        None
                    } else {
                        Some(g.to_uppercase())
                    }
                } else {
                    None
                };
                let message = if tokens.len() > 3 {
                    Some(tokens[3..].join(" "))
                } else {
                    None
                };
                let room_id = if let Some(r) = self.ctx.room.as_ref() {
                    r.to_string()
                } else {
                    self.push_plain("(builder) no current room; set one with /room");
                    return;
                };
                let Some(parser) = self.parser.as_mut() else {
                    self.push_plain("(builder) parser not loaded");
                    return;
                };
                let Some(world) = parser.world.as_mut() else {
                    self.push_plain("(builder) no world loaded; use /newworld first");
                    return;
                };
                match add_guarded_exit(
                    world,
                    &room_id,
                    dir,
                    dest,
                    guard_flag.as_deref(),
                    message.as_deref(),
                    !oneway,
                    false,
                    None,
                    self.engine.tick,
                    cmdline,
                ) {
                    Ok(change) => {
                        self.refresh_world_views();
                        self.persist_builder_change(Some(&change));
                        self.push_plain(format!("(builder) {}", change.effect));
                    }
                    Err(e) => self.push_plain(format!("(builder) {}", e)),
                }
            }
            "door" => {
                if rest.is_empty() {
                    self.push_plain("Usage: /door <dir> [dest] <locked|unlocked> [flag] [key]");
                    return;
                }
                let tokens: Vec<&str> = rest.split_whitespace().collect();
                if tokens.len() < 2 {
                    self.push_plain("Usage: /door <dir> [dest] <locked|unlocked> [flag] [key]");
                    return;
                }
                let dir = tokens[0];
                let mut dest: Option<&str> = None;
                let mut lock_state: Option<&str> = None;
                let mut idx = 1;
                while idx < tokens.len() {
                    let t = tokens[idx];
                    if t.eq_ignore_ascii_case("locked") || t.eq_ignore_ascii_case("unlocked") {
                        lock_state = Some(t);
                        idx += 1;
                        break;
                    }
                    if dest.is_none() {
                        dest = Some(t);
                    }
                    idx += 1;
                }
                if lock_state.is_none() {
                    self.push_plain("Usage: /door <dir> [dest] <locked|unlocked> [flag] [key]");
                    return;
                }
                let guard_flag = if idx < tokens.len() {
                    let g = tokens[idx].trim();
                    idx += 1;
                    if g.eq_ignore_ascii_case("none") || g.is_empty() {
                        None
                    } else {
                        Some(g.to_uppercase())
                    }
                } else {
                    None
                };
                let key_hint = if idx < tokens.len() {
                    Some(tokens[idx])
                } else {
                    None
                };
                let locked = lock_state
                    .map(|s| s.eq_ignore_ascii_case("locked"))
                    .unwrap_or(false);
                let room_id = if let Some(r) = self.ctx.room.as_ref() {
                    r.to_string()
                } else {
                    self.push_plain("(builder) no current room; set one with /room");
                    return;
                };
                let Some(parser) = self.parser.as_mut() else {
                    self.push_plain("(builder) parser not loaded");
                    return;
                };
                let Some(world) = parser.world.as_mut() else {
                    self.push_plain("(builder) no world loaded; use /newworld first");
                    return;
                };
                match add_door(
                    world,
                    &room_id,
                    dir,
                    dest,
                    locked,
                    guard_flag.as_deref(),
                    key_hint,
                    self.engine.tick,
                    cmdline,
                ) {
                    Ok(change) => {
                        self.refresh_world_views();
                        self.persist_builder_change(Some(&change));
                        self.push_plain(format!("(builder) {}", change.effect));
                    }
                    Err(e) => self.push_plain(format!("(builder) {}", e)),
                }
            }
            "counter" => {
                if rest.is_empty() {
                    self.push_plain("Usage: /counter <name> <value|+N|-N> [drain N]");
                    return;
                }
                let tokens: Vec<&str> = rest.split_whitespace().collect();
                if tokens.len() < 2 {
                    self.push_plain("Usage: /counter <name> <value|+N|-N> [drain N]");
                    return;
                }
                let name = tokens[0];
                let val_token = tokens[1];
                let mut drain_set: Option<i64> = None;
                if tokens.len() >= 4 && tokens[2].eq_ignore_ascii_case("drain") {
                    match tokens[3].parse::<i64>() {
                        Ok(d) => drain_set = Some(d),
                        Err(_) => {
                            self.push_plain("(builder) drain must be numeric");
                            return;
                        }
                    }
                } else if tokens.len() == 3 && tokens[2].eq_ignore_ascii_case("drain") {
                    self.push_plain("Usage: /counter <name> <value|+N|-N> [drain N]");
                    return;
                }
                let op = if val_token.starts_with('+') || val_token.starts_with('-') {
                    match val_token.parse::<i64>() {
                        Ok(delta) => CounterOp::Delta(delta),
                        Err(_) => {
                            self.push_plain("(builder) counter value must be numeric");
                            return;
                        }
                    }
                } else {
                    match val_token.parse::<i64>() {
                        Ok(v) => CounterOp::Set(v),
                        Err(_) => {
                            self.push_plain("(builder) counter value must be numeric");
                            return;
                        }
                    }
                };
                let Some(parser) = self.parser.as_mut() else {
                    self.push_plain("(builder) parser not loaded");
                    return;
                };
                let Some(world) = parser.world.as_mut() else {
                    self.push_plain("(builder) no world loaded; use /newworld first");
                    return;
                };
                let tick = self.engine.tick;
                match update_counter(world, &mut self.engine, name, op, drain_set, tick, cmdline) {
                    Ok(change) => {
                        self.persist_builder_change(Some(&change));
                        self.push_plain(format!("(builder) {}", change.effect));
                    }
                    Err(e) => self.push_plain(format!("(builder) {}", e)),
                }
            }
            "npc" => {
                if rest.is_empty() {
                    self.push_plain("Usage: /npc <name>, <description>[, room]");
                    return;
                }
                let mut parts = rest.splitn(3, ',').map(|s| s.trim());
                let name = parts.next().unwrap_or("");
                let desc = parts.next().unwrap_or("");
                let room = parts.next();
                if name.is_empty() || desc.is_empty() {
                    self.push_plain("Usage: /npc <name>, <description>[, room]");
                    return;
                }
                let location = if let Some(r) = room {
                    let t = r.trim();
                    if t.is_empty() {
                        None
                    } else {
                        Some(normalize_id(t))
                    }
                } else {
                    self.ctx.room.clone()
                };
                if location.is_none() {
                    self.push_plain("(builder) no current room; set one with /room");
                    return;
                }
                let Some(parser) = self.parser.as_mut() else {
                    self.push_plain("(builder) parser not loaded");
                    return;
                };
                let Some(world) = parser.world.as_mut() else {
                    self.push_plain("(builder) no world loaded; use /newworld first");
                    return;
                };
                match upsert_npc(
                    world,
                    name,
                    desc,
                    location.as_deref(),
                    self.engine.tick,
                    cmdline,
                ) {
                    Ok(change) => {
                        self.persist_builder_change(Some(&change));
                        self.push_plain(format!("(builder) {}", change.effect));
                    }
                    Err(e) => self.push_plain(format!("(builder) {}", e)),
                }
            }
            "npcplace" => {
                if rest.is_empty() {
                    self.push_plain("Usage: /npcplace <npc> [room]");
                    return;
                }
                let mut parts = rest.split_whitespace();
                let npc_id = parts.next().unwrap_or("");
                let room = parts
                    .next()
                    .map(|r| normalize_id(r))
                    .or_else(|| self.ctx.room.clone());
                if room.is_none() {
                    self.push_plain("(builder) no room specified; set one with /room");
                    return;
                }
                let Some(parser) = self.parser.as_mut() else {
                    self.push_plain("(builder) parser not loaded");
                    return;
                };
                let Some(world) = parser.world.as_mut() else {
                    self.push_plain("(builder) no world loaded; use /newworld first");
                    return;
                };
                match upsert_npc(
                    world,
                    npc_id,
                    "",
                    room.as_deref(),
                    self.engine.tick,
                    cmdline,
                ) {
                    Ok(change) => {
                        self.persist_builder_change(Some(&change));
                        self.push_plain(format!("(builder) {}", change.effect));
                    }
                    Err(e) => self.push_plain(format!("(builder) {}", e)),
                }
            }
            "npcnode" => {
                if rest.is_empty() {
                    self.push_plain(
                        "Usage: /npcnode <npc> <node> \"text\" [requires] [effects] [repeat]",
                    );
                    return;
                }
                let mut parts = rest.splitn(3, char::is_whitespace);
                let npc_id = parts.next().unwrap_or("");
                let node_id = parts.next().unwrap_or("");
                let tail = parts.next().unwrap_or("").trim_start();
                if npc_id.is_empty() || node_id.is_empty() || tail.is_empty() {
                    self.push_plain(
                        "Usage: /npcnode <npc> <node> \"text\" [requires] [effects] [repeat]",
                    );
                    return;
                }
                let (text, tail_after_text) = if tail.starts_with('"') {
                    if let Some((txt, rest_tail)) = tail[1..].split_once('"') {
                        (txt.to_string(), rest_tail.trim())
                    } else {
                        self.push_plain("(builder) unmatched quote in node text");
                        return;
                    }
                } else {
                    let mut split = tail.splitn(2, char::is_whitespace);
                    let txt = split.next().unwrap_or("").to_string();
                    let rest_tail = split.next().unwrap_or("").trim();
                    (txt, rest_tail)
                };
                let mut tail_tokens = tail_after_text.split_whitespace();
                let requires_raw = tail_tokens.next().unwrap_or("");
                let changes_raw = tail_tokens.next().unwrap_or("");
                let repeatable = tail_tokens.any(|t| t.eq_ignore_ascii_case("repeat"));
                let requires = parse_flag_list(requires_raw);
                let ParsedEffects {
                    set_flags,
                    clear_flags,
                    counter_changes,
                    give_items,
                    take_items,
                    credits,
                } = Self::parse_effect_tokens(changes_raw);
                let Some(parser) = self.parser.as_mut() else {
                    self.push_plain("(builder) parser not loaded");
                    return;
                };
                let Some(world) = parser.world.as_mut() else {
                    self.push_plain("(builder) no world loaded; use /newworld first");
                    return;
                };
                match add_npc_node(
                    world,
                    npc_id,
                    node_id,
                    &text,
                    requires,
                    set_flags,
                    clear_flags,
                    counter_changes,
                    give_items,
                    take_items,
                    credits,
                    repeatable,
                    self.engine.tick,
                    cmdline,
                ) {
                    Ok(change) => {
                        self.persist_builder_change(Some(&change));
                        self.push_plain(format!("(builder) {}", change.effect));
                    }
                    Err(e) => self.push_plain(format!("(builder) {}", e)),
                }
            }
            "npclink" => {
                if rest.is_empty() {
                    self.push_plain("Usage: /npclink <npc> <from> <to> \"prompt\" [requires] [effects] [hidden]");
                    return;
                }
                let mut parts = rest.splitn(4, char::is_whitespace);
                let npc_id = parts.next().unwrap_or("");
                let from = parts.next().unwrap_or("");
                let to = parts.next().unwrap_or("");
                let tail = parts.next().unwrap_or("").trim_start();
                if npc_id.is_empty() || from.is_empty() || to.is_empty() || tail.is_empty() {
                    self.push_plain("Usage: /npclink <npc> <from> <to> \"prompt\" [requires] [effects] [hidden]");
                    return;
                }
                let (prompt, tail_after_prompt) = if tail.starts_with('"') {
                    if let Some((txt, rest_tail)) = tail[1..].split_once('"') {
                        (txt.to_string(), rest_tail.trim())
                    } else {
                        self.push_plain("(builder) unmatched quote in link prompt");
                        return;
                    }
                } else {
                    let mut split = tail.splitn(2, char::is_whitespace);
                    let txt = split.next().unwrap_or("").to_string();
                    let rest_tail = split.next().unwrap_or("").trim();
                    (txt, rest_tail)
                };
                let mut tail_tokens = tail_after_prompt.split_whitespace();
                let requires_raw = tail_tokens.next().unwrap_or("");
                let changes_raw = tail_tokens.next().unwrap_or("");
                let hidden = tail_tokens.any(|t| t.eq_ignore_ascii_case("hidden"));
                let requires = parse_flag_list(requires_raw);
                let ParsedEffects {
                    set_flags,
                    clear_flags,
                    counter_changes,
                    give_items,
                    take_items,
                    credits,
                } = Self::parse_effect_tokens(changes_raw);
                let Some(parser) = self.parser.as_mut() else {
                    self.push_plain("(builder) parser not loaded");
                    return;
                };
                let Some(world) = parser.world.as_mut() else {
                    self.push_plain("(builder) no world loaded; use /newworld first");
                    return;
                };
                match add_npc_link(
                    world,
                    npc_id,
                    from,
                    to,
                    &prompt,
                    requires,
                    set_flags,
                    clear_flags,
                    counter_changes,
                    give_items,
                    take_items,
                    credits,
                    hidden,
                    self.engine.tick,
                    cmdline,
                ) {
                    Ok(change) => {
                        self.persist_builder_change(Some(&change));
                        self.push_plain(format!("(builder) {}", change.effect));
                    }
                    Err(e) => self.push_plain(format!("(builder) {}", e)),
                }
            }
            "npcsay" => {
                if rest.is_empty() {
                    self.push_plain("Usage: /npcsay <npc_id> [node_id]");
                    return;
                }
                let mut parts = rest.split_whitespace();
                let npc_id = parts.next().unwrap_or("");
                let node_id = parts.next();
                self.start_dialog_with(npc_id, node_id);
            }
            "npcchoose" => {
                if rest.is_empty() {
                    self.push_plain("Usage: /npcchoose <index>");
                    return;
                }
                let idx: usize = match rest.trim().parse::<usize>() {
                    Ok(n) if n > 0 => n - 1,
                    _ => {
                        self.push_plain("(dialog) index must be a positive number");
                        return;
                    }
                };
                self.choose_dialog_option_idx(idx);
            }
            "name" => {
                if rest.is_empty() {
                    self.push_plain("Usage: /name <new_room_id>");
                    return;
                }
                let room_id = if let Some(r) = self.ctx.room.as_ref() {
                    r.to_string()
                } else {
                    self.push_plain("(builder) no current room; set one with /room");
                    return;
                };
                let Some(parser) = self.parser.as_mut() else {
                    self.push_plain("(builder) parser not loaded");
                    return;
                };
                let Some(world) = parser.world.as_mut() else {
                    self.push_plain("(builder) no world loaded; use /newworld first");
                    return;
                };
                match rename_room(world, &room_id, rest, self.engine.tick, cmdline) {
                    Ok(change) => {
                        self.ctx.room = Some(normalize_id(rest));
                        self.refresh_world_views();
                        self.persist_builder_change(Some(&change));
                        self.push_plain(format!(
                            "(builder) renamed room {} -> {}",
                            room_id,
                            normalize_id(rest)
                        ));
                        self.reload_room_media(ctx);
                        self.log_room_description(true, false, false);
                    }
                    Err(e) => self.push_plain(format!("(builder) {}", e)),
                }
            }
            "ldesc" => {
                if rest.is_empty() {
                    self.push_plain("Usage: /ldesc <long description text>");
                    return;
                }
                let room_id = if let Some(r) = self.ctx.room.as_ref() {
                    r.to_string()
                } else {
                    self.push_plain("(builder) no current room; set one with /room");
                    return;
                };
                let Some(parser) = self.parser.as_mut() else {
                    self.push_plain("(builder) parser not loaded");
                    return;
                };
                let Some(world) = parser.world.as_mut() else {
                    self.push_plain("(builder) no world loaded; use /newworld first");
                    return;
                };
                match set_room_ldesc(world, &room_id, rest, self.engine.tick, cmdline) {
                    Ok(change) => {
                        self.refresh_world_views();
                        self.persist_builder_change(Some(&change));
                        self.push_plain(format!("(builder) updated description for {}", room_id));
                        self.log_room_description(true, false, false);
                    }
                    Err(e) => self.push_plain(format!("(builder) {}", e)),
                }
            }
            "sdesc" => {
                if rest.is_empty() {
                    self.push_plain("Usage: /sdesc <short description text>");
                    return;
                }
                let room_id = if let Some(r) = self.ctx.room.as_ref() {
                    r.to_string()
                } else {
                    self.push_plain("(builder) no current room; set one with /room");
                    return;
                };
                let Some(parser) = self.parser.as_mut() else {
                    self.push_plain("(builder) parser not loaded");
                    return;
                };
                let Some(world) = parser.world.as_mut() else {
                    self.push_plain("(builder) no world loaded; use /newworld first");
                    return;
                };
                match set_room_sdesc(world, &room_id, rest, self.engine.tick, cmdline) {
                    Ok(change) => {
                        self.refresh_world_views();
                        self.persist_builder_change(Some(&change));
                        self.push_plain(format!(
                            "(builder) updated short description for {}",
                            room_id
                        ));
                        self.log_room_description(true, false, false);
                    }
                    Err(e) => self.push_plain(format!("(builder) {}", e)),
                }
            }
            "note" => {
                if rest.is_empty() {
                    self.push_plain("Usage: /note <text>");
                    return;
                }
                let room_id = if let Some(r) = self.ctx.room.as_ref() {
                    r.to_string()
                } else {
                    self.push_plain("(builder) no current room; set one with /room");
                    return;
                };
                let Some(parser) = self.parser.as_mut() else {
                    self.push_plain("(builder) parser not loaded");
                    return;
                };
                let Some(world) = parser.world.as_mut() else {
                    self.push_plain("(builder) no world loaded; use /newworld first");
                    return;
                };
                match set_room_note(world, &room_id, rest, self.engine.tick, cmdline) {
                    Ok(change) => {
                        self.persist_builder_change(Some(&change));
                        self.push_plain(format!("(builder) noted {}", room_id));
                    }
                    Err(e) => self.push_plain(format!("(builder) {}", e)),
                }
            }
            "ambient" => {
                if rest.is_empty() {
                    self.push_plain("Usage: /ambient <text>");
                    return;
                }
                let room_id = if let Some(r) = self.ctx.room.as_ref() {
                    r.to_string()
                } else {
                    self.push_plain("(builder) no current room; set one with /room");
                    return;
                };
                let Some(parser) = self.parser.as_mut() else {
                    self.push_plain("(builder) parser not loaded");
                    return;
                };
                let Some(world) = parser.world.as_mut() else {
                    self.push_plain("(builder) no world loaded; use /newworld first");
                    return;
                };
                match set_room_ambient(world, &room_id, rest, self.engine.tick, cmdline) {
                    Ok(change) => {
                        self.refresh_world_views();
                        self.persist_builder_change(Some(&change));
                        self.push_plain(format!("(builder) set ambient for {}", room_id));
                    }
                    Err(e) => self.push_plain(format!("(builder) {}", e)),
                }
            }
            "variant" => {
                if rest.is_empty() {
                    self.push_plain("Usage: /variant <flag|none> <text>");
                    return;
                }
                let mut parts = rest.splitn(2, char::is_whitespace);
                let guard_raw = parts.next().unwrap_or("");
                let text_raw = parts.next().unwrap_or("").trim_start();
                if guard_raw.is_empty() || text_raw.is_empty() {
                    self.push_plain("Usage: /variant <flag|none> <text>");
                    return;
                }
                let text_clean = if let Some(stripped) = text_raw.strip_prefix('"') {
                    if let Some((body, _)) = stripped.split_once('"') {
                        body.to_string()
                    } else {
                        text_raw.to_string()
                    }
                } else {
                    text_raw.to_string()
                };
                let room_id = if let Some(r) = self.ctx.room.as_ref() {
                    r.to_string()
                } else {
                    self.push_plain("(builder) no current room; set one with /room");
                    return;
                };
                let Some(parser) = self.parser.as_mut() else {
                    self.push_plain("(builder) parser not loaded");
                    return;
                };
                let Some(world) = parser.world.as_mut() else {
                    self.push_plain("(builder) no world loaded; use /newworld first");
                    return;
                };
                match add_room_variant(
                    world,
                    &room_id,
                    Some(guard_raw),
                    &text_clean,
                    self.engine.tick,
                    cmdline,
                ) {
                    Ok(change) => {
                        self.refresh_world_views();
                        self.persist_builder_change(Some(&change));
                        self.push_plain(format!("(builder) {}", change.effect));
                    }
                    Err(e) => self.push_plain(format!("(builder) {}", e)),
                }
            }
            "start" => {
                let target = if rest.is_empty() {
                    if let Some(r) = self.ctx.room.as_ref() {
                        r.to_string()
                    } else {
                        self.push_plain("Usage: /start <room>");
                        return;
                    }
                } else {
                    normalize_room_id(rest)
                };
                let Some(parser) = self.parser.as_mut() else {
                    self.push_plain("(builder) parser not loaded");
                    return;
                };
                let Some(world) = parser.world.as_mut() else {
                    self.push_plain("(builder) no world loaded; use /newworld first");
                    return;
                };
                match set_start_room(world, &target, self.engine.tick, cmdline) {
                    Ok(change) => {
                        self.persist_builder_change(Some(&change));
                        self.push_plain(format!("(builder) start room set to {}", target));
                    }
                    Err(e) => self.push_plain(format!("(builder) {}", e)),
                }
            }
            "objname" => {
                if rest.is_empty() {
                    self.push_plain("Usage: /objname <name>, <newname>");
                    return;
                }
                let mut parts = rest.splitn(2, ',').map(|s| s.trim());
                let old_name = parts.next().unwrap_or("");
                let new_name = parts.next().unwrap_or("");
                if old_name.is_empty() || new_name.is_empty() {
                    self.push_plain("Usage: /objname <name>, <newname>");
                    return;
                }
                let Some(parser) = self.parser.as_mut() else {
                    self.push_plain("(builder) parser not loaded");
                    return;
                };
                let Some(world) = parser.world.as_mut() else {
                    self.push_plain("(builder) no world loaded; use /newworld first");
                    return;
                };
                match rename_object(world, old_name, new_name, self.engine.tick, cmdline) {
                    Ok(change) => {
                        self.refresh_world_views();
                        self.persist_builder_change(Some(&change));
                        self.push_plain(format!(
                            "(builder) renamed object {} -> {}",
                            normalize_id(old_name),
                            normalize_id(new_name)
                        ));
                    }
                    Err(e) => self.push_plain(format!("(builder) {}", e)),
                }
            }
            "moveobj" => {
                if rest.is_empty() {
                    self.push_plain("Usage: /moveobj <object> <room|inventory|object>");
                    return;
                }
                let mut parts = rest.split_whitespace();
                let obj_id = parts.next().unwrap_or("");
                let dest = parts.next().unwrap_or("");
                if obj_id.is_empty() || dest.is_empty() {
                    self.push_plain("Usage: /moveobj <object> <room|inventory|object>");
                    return;
                }
                let Some(parser) = self.parser.as_mut() else {
                    self.push_plain("(builder) parser not loaded");
                    return;
                };
                let Some(world) = parser.world.as_mut() else {
                    self.push_plain("(builder) no world loaded; use /newworld first");
                    return;
                };
                match move_object(world, obj_id, dest, self.engine.tick, cmdline) {
                    Ok(change) => {
                        self.refresh_world_views();
                        self.persist_builder_change(Some(&change));
                        self.push_plain(format!(
                            "(builder) moved {} -> {}",
                            normalize_id(obj_id),
                            normalize_id(dest)
                        ));
                    }
                    Err(e) => self.push_plain(format!("(builder) {}", e)),
                }
            }
            "objflags" => {
                if rest.is_empty() {
                    self.push_plain("Usage: /objflags <object> +FLAG -FLAG ...");
                    return;
                }
                let mut parts = rest.split_whitespace();
                let obj_id = parts.next().unwrap_or("");
                if obj_id.is_empty() {
                    self.push_plain("Usage: /objflags <object> +FLAG -FLAG ...");
                    return;
                }
                let mut adds: Vec<String> = Vec::new();
                let mut removes: Vec<String> = Vec::new();
                for tok in parts {
                    if let Some(flag) = tok.strip_prefix('+') {
                        adds.push(flag.to_string());
                    } else if let Some(flag) = tok.strip_prefix('-') {
                        removes.push(flag.to_string());
                    } else if let Some(flag) = tok.strip_prefix('!') {
                        removes.push(flag.to_string());
                    } else {
                        adds.push(tok.to_string());
                    }
                }
                if adds.is_empty() && removes.is_empty() {
                    self.push_plain("Usage: /objflags <object> +FLAG -FLAG ...");
                    return;
                }
                let Some(parser) = self.parser.as_mut() else {
                    self.push_plain("(builder) parser not loaded");
                    return;
                };
                let Some(world) = parser.world.as_mut() else {
                    self.push_plain("(builder) no world loaded; use /newworld first");
                    return;
                };
                match update_object_flags(world, obj_id, &adds, &removes, self.engine.tick, cmdline)
                {
                    Ok(change) => {
                        self.refresh_world_views();
                        self.persist_builder_change(Some(&change));
                        self.push_plain(format!(
                            "(builder) flags updated for {}",
                            normalize_id(obj_id)
                        ));
                    }
                    Err(e) => self.push_plain(format!("(builder) {}", e)),
                }
            }
            "cloneobj" => {
                if rest.is_empty() {
                    self.push_plain("Usage: /cloneobj <source> <new> [dest]");
                    return;
                }
                let mut parts = rest.split_whitespace();
                let src = parts.next().unwrap_or("");
                let dst = parts.next().unwrap_or("");
                let dest = parts.next();
                if src.is_empty() || dst.is_empty() {
                    self.push_plain("Usage: /cloneobj <source> <new> [dest]");
                    return;
                }
                let Some(parser) = self.parser.as_mut() else {
                    self.push_plain("(builder) parser not loaded");
                    return;
                };
                let Some(world) = parser.world.as_mut() else {
                    self.push_plain("(builder) no world loaded; use /newworld first");
                    return;
                };
                match clone_object(world, src, dst, dest, self.engine.tick, cmdline) {
                    Ok(change) => {
                        self.refresh_world_views();
                        self.persist_builder_change(Some(&change));
                        self.push_plain(format!(
                            "(builder) cloned {} -> {}",
                            normalize_id(src),
                            normalize_id(dst)
                        ));
                    }
                    Err(e) => self.push_plain(format!("(builder) {}", e)),
                }
            }
            "cloneroom" => {
                if rest.is_empty() {
                    self.push_plain("Usage: /cloneroom <source> <new>");
                    return;
                }
                let mut parts = rest.split_whitespace();
                let src = parts.next().unwrap_or("");
                let dst = parts.next().unwrap_or("");
                if src.is_empty() || dst.is_empty() {
                    self.push_plain("Usage: /cloneroom <source> <new>");
                    return;
                }
                let Some(parser) = self.parser.as_mut() else {
                    self.push_plain("(builder) parser not loaded");
                    return;
                };
                let Some(world) = parser.world.as_mut() else {
                    self.push_plain("(builder) no world loaded; use /newworld first");
                    return;
                };
                match clone_room(world, src, dst, self.engine.tick, cmdline) {
                    Ok(change) => {
                        self.refresh_world_views();
                        self.persist_builder_change(Some(&change));
                        self.push_plain(format!("(builder) {}", change.effect));
                    }
                    Err(e) => self.push_plain(format!("(builder) {}", e)),
                }
            }
            "exportroom" => {
                if rest.is_empty() {
                    self.push_plain("Usage: /exportroom <room> <path>");
                    return;
                }
                let mut parts = rest.split_whitespace();
                let room = parts.next().unwrap_or("");
                let path_raw = parts.next();
                if room.is_empty() || path_raw.is_none() {
                    self.push_plain("Usage: /exportroom <room> <path>");
                    return;
                }
                let Some(parser) = self.parser.as_mut() else {
                    self.push_plain("(builder) parser not loaded");
                    return;
                };
                let Some(world) = parser.world.as_mut() else {
                    self.push_plain("(builder) no world loaded; use /newworld first");
                    return;
                };
                let path = PathBuf::from(path_raw.unwrap());
                match export_room(world, room, &path, self.engine.tick, cmdline) {
                    Ok(change) => {
                        self.persist_builder_change(Some(&change));
                        self.push_plain(format!("(builder) {}", change.effect));
                    }
                    Err(e) => self.push_plain(format!("(builder) {}", e)),
                }
            }
            "importr" => {
                if rest.is_empty() {
                    self.push_plain("Usage: /importr <path> [new_id]");
                    return;
                }
                let mut parts = rest.split_whitespace();
                let path_raw = parts.next().unwrap_or("");
                let new_id = parts.next();
                let Some(parser) = self.parser.as_mut() else {
                    self.push_plain("(builder) parser not loaded");
                    return;
                };
                let Some(world) = parser.world.as_mut() else {
                    self.push_plain("(builder) no world loaded; use /newworld first");
                    return;
                };
                let path = PathBuf::from(path_raw);
                match import_room(world, &path, new_id, self.engine.tick, cmdline) {
                    Ok(change) => {
                        self.refresh_world_views();
                        self.persist_builder_change(Some(&change));
                        self.push_plain(format!("(builder) {}", change.effect));
                    }
                    Err(e) => self.push_plain(format!("(builder) {}", e)),
                }
            }
            "nobj" => {
                if rest.is_empty() {
                    self.push_plain("Usage: /nobj <name>, <description>[, container]");
                    return;
                }
                let mut parts = rest.splitn(3, ',').map(|s| s.trim());
                let name = parts.next().unwrap_or("");
                let desc = parts.next().unwrap_or("");
                let trailing = parts.next().unwrap_or("");
                if name.is_empty() || desc.is_empty() {
                    self.push_plain("Usage: /nobj <name>, <description>[, container]");
                    return;
                }
                let is_container = trailing.to_ascii_lowercase().contains("container");
                let room_id = if let Some(r) = self.ctx.room.as_ref() {
                    r.to_string()
                } else {
                    self.push_plain("(builder) no current room; set one with /room");
                    return;
                };
                let Some(parser) = self.parser.as_mut() else {
                    self.push_plain("(builder) parser not loaded");
                    return;
                };
                let Some(world) = parser.world.as_mut() else {
                    self.push_plain("(builder) no world loaded; use /newworld first");
                    return;
                };
                match add_object_to_room(
                    world,
                    &room_id,
                    name,
                    desc,
                    is_container,
                    self.engine.tick,
                    cmdline,
                ) {
                    Ok(change) => {
                        self.refresh_world_views();
                        self.persist_builder_change(Some(&change));
                        self.push_plain(format!(
                            "(builder) added {} to {}",
                            normalize_id(name),
                            room_id
                        ));
                    }
                    Err(e) => self.push_plain(format!("(builder) {}", e)),
                }
            }
            "action" => {
                if rest.is_empty() {
                    self.push_plain("Usage: /action <command> <flags_required> <flags_changed>");
                    return;
                }
                let (command, tail) = if rest.starts_with('"') {
                    if let Some((cmd_part, tail_part)) = rest[1..].split_once('"') {
                        (cmd_part.trim(), tail_part.trim())
                    } else {
                        self.push_plain("(builder) unmatched quote in command");
                        return;
                    }
                } else {
                    let mut split = rest.splitn(2, char::is_whitespace);
                    let cmd_part = split.next().unwrap_or("").trim();
                    let tail_part = split.next().unwrap_or("").trim();
                    (cmd_part, tail_part)
                };
                if command.is_empty() {
                    self.push_plain("(builder) command cannot be empty");
                    return;
                }
                let mut tail_parts = tail.splitn(2, char::is_whitespace);
                let requires_raw = tail_parts.next().unwrap_or("");
                let changes_raw = tail_parts.next().unwrap_or("");
                let room_id = if let Some(r) = self.ctx.room.as_ref() {
                    r.to_string()
                } else {
                    self.push_plain("(builder) no current room; set one with /room");
                    return;
                };
                let Some(parser) = self.parser.as_mut() else {
                    self.push_plain("(builder) parser not loaded");
                    return;
                };
                let Some(world) = parser.world.as_mut() else {
                    self.push_plain("(builder) no world loaded; use /newworld first");
                    return;
                };
                let mut action = BuilderAction::default();
                action.command = command.to_string();
                action.requires_flags = parse_flag_list(requires_raw);
                let (set_flags, clear_flags) = parse_flag_changes(changes_raw);
                action.set_flags = set_flags;
                action.clear_flags = clear_flags;
                match add_room_action(world, &room_id, action, self.engine.tick, cmdline) {
                    Ok(change) => {
                        self.refresh_world_views();
                        self.persist_builder_change(Some(&change));
                        self.push_plain(format!("(builder) added action in {}", room_id));
                    }
                    Err(e) => self.push_plain(format!("(builder) {}", e)),
                }
            }
            "flag" => {
                if rest.is_empty() {
                    self.push_plain("Usage: /flag [+]FLAG|-FLAG [...]");
                    return;
                }
                let mut any = false;
                let mut tokens: Vec<(bool, String)> = Vec::new();
                for tok in rest.split_whitespace() {
                    let (value, name_raw) = if let Some(name) = tok.strip_prefix('-') {
                        (false, name)
                    } else if let Some(name) = tok.strip_prefix('!') {
                        (false, name)
                    } else if let Some(name) = tok.strip_prefix('+') {
                        (true, name)
                    } else {
                        (true, tok)
                    };
                    if name_raw.trim().is_empty() {
                        continue;
                    }
                    any = true;
                    tokens.push((value, name_raw.to_string()));
                }
                if !any {
                    self.push_plain("Usage: /flag [+]FLAG|-FLAG [...]");
                    return;
                }
                let mut changes: Vec<BuilderChange> = Vec::new();
                let mut errs: Vec<String> = Vec::new();
                {
                    let Some(parser) = self.parser.as_mut() else {
                        self.push_plain("(builder) parser not loaded");
                        return;
                    };
                    let Some(world) = parser.world.as_mut() else {
                        self.push_plain("(builder) no world loaded; use /newworld first");
                        return;
                    };
                    let tick = self.engine.tick;
                    for (value, name_raw) in tokens.iter() {
                        match set_global_flag(
                            world,
                            &mut self.engine,
                            name_raw,
                            *value,
                            tick,
                            cmdline,
                        ) {
                            Ok(change) => changes.push(change),
                            Err(e) => errs.push(e),
                        }
                    }
                }
                for e in errs {
                    self.push_plain(format!("(builder) {}", e));
                }
                for (idx, change) in changes.iter().enumerate() {
                    self.persist_builder_change(Some(change));
                    if let Some((value, name_raw)) = tokens.get(idx) {
                        self.push_plain(format!(
                            "(builder) flag {} {}",
                            normalize_id(name_raw),
                            if *value { "set" } else { "cleared" }
                        ));
                    } else {
                        self.push_plain("(builder) flag updated");
                    }
                }
            }
            "lint" => {
                if let Some(world) = self.parser.as_ref().and_then(|p| p.world.as_ref()) {
                    let issues = lint_world_with_assets(world, self.manifest.as_ref());
                    if issues.is_empty() {
                        self.push_plain("(lint) no issues found");
                    } else {
                        self.push_plain(format!("(lint) {} issue(s):", issues.len()));
                        for issue in issues {
                            self.push_plain(format!("- {}", format_lint_issue(&issue)));
                        }
                    }
                } else {
                    self.push_plain("(lint) no world loaded; use /newworld first");
                }
            }
            "saveworld" => {
                let target_path = if rest.is_empty() {
                    self.builder_world_path.clone()
                } else {
                    let pb = PathBuf::from(rest);
                    self.builder_world_path = Some(pb.clone());
                    if self.builder_log_path.is_none() {
                        self.builder_log_path = Some(pb.with_extension("log.jsonl"));
                    }
                    Some(pb)
                };
                if let Some(path) = target_path {
                    if let Some(dir) = path.parent() {
                        let _ = fs::create_dir_all(dir);
                    }
                    if let Some(world) = self.parser.as_ref().and_then(|p| p.world.as_ref()) {
                        match save_world(&path, world) {
                            Ok(_) => self
                                .push_plain(format!("(builder) saved world to {}", path.display())),
                            Err(e) => self.push_plain(format!("(builder) save failed: {}", e)),
                        }
                    } else {
                        self.push_plain("(builder) no world loaded; use /newworld first");
                    }
                } else {
                    self.push_plain("Usage: /saveworld <path> (or set via /newworld)");
                }
            }
            "rebuild" => {
                if self.rebuild_rx.is_some() {
                    self.push_plain("(builder) rebuild already running");
                    return;
                }
                let Some(world_path) = self.builder_world_path.clone() else {
                    self.push_plain("(builder) no world loaded; use /newworld first");
                    return;
                };
                let Some(world_snapshot) = self.parser.as_ref().and_then(|p| p.world.clone())
                else {
                    self.push_plain("(builder) no world loaded; use /newworld first");
                    return;
                };
                let (tx, rx) = mpsc::channel::<String>();
                let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join("..")
                    .join("..");
                let mut world_label = world_snapshot.game.trim().to_string();
                if world_label.is_empty() {
                    world_label = world_path
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .unwrap_or("builder")
                        .to_string();
                }
                let seed_path =
                    world_path.with_file_name(format!("{}_vocab_seed.json", world_label));
                self.rebuild_rx = Some(rx);
                self.push_plain("(builder) rebuild started");
                std::thread::spawn(move || {
                    let send = |msg: String, tx: &Sender<String>| {
                        let _ = tx.send(msg);
                    };
                    if let Some(dir) = world_path.parent() {
                        let _ = fs::create_dir_all(dir);
                    }
                    if let Err(e) = save_world(&world_path, &world_snapshot) {
                        let _ = tx.send(format!("__DONE_ERR__save failed: {}", e));
                        return;
                    }
                    let seed = match write_vocab_seed(&world_snapshot, &seed_path) {
                        Ok(seed) => seed,
                        Err(e) => {
                            let _ = tx.send(format!("__DONE_ERR__vocab seed failed: {}", e));
                            return;
                        }
                    };
                    send(
                        format!("(builder) wrote vocab seed {}", seed_path.display()),
                        &tx,
                    );
                    let seed_arg = seed_path
                        .canonicalize()
                        .unwrap_or_else(|_| seed_path.clone());
                    let enrich_script = repo_root.join("enrich_vocab.py");
                    if enrich_script.exists() {
                        send("(builder) running enrich_vocab ...".into(), &tx);
                        let mut cmd = Command::new("python");
                        cmd.current_dir(&repo_root);
                        cmd.arg(&enrich_script);
                        cmd.arg("--vocab");
                        cmd.arg(repo_root.join("zork1_vocab.json"));
                        cmd.arg(repo_root.join("zork2_vocab.json"));
                        cmd.arg(repo_root.join("zork3_vocab.json"));
                        cmd.arg(seed_arg.clone());
                        cmd.arg("--out-dir");
                        cmd.arg(repo_root.join("out"));
                        cmd.arg("--use-local-conceptnet");
                        cmd.arg("--no-embeddings");
                        if let Err(e) =
                            run_command_stream_channel(&mut cmd, "enrich_vocab", tx.clone())
                        {
                            let _ = tx.send(format!("__DONE_ERR__{}", e));
                            return;
                        }
                        send("(builder) enrich_vocab completed".into(), &tx);
                    } else {
                        send(
                            format!(
                                "(builder) enrich_vocab.py not found at {}",
                                enrich_script.display()
                            ),
                            &tx,
                        );
                    }
                    match merge_seed_into_parser_files(&seed, &repo_root) {
                        Ok(p) => send(
                            format!("(builder) merged parser vocab -> {}", p.display()),
                            &tx,
                        ),
                        Err(e) => {
                            let _ = tx
                                .send(format!("__DONE_ERR__merge into parser_only failed: {}", e));
                            return;
                        }
                    }
                    let canonical_world = repo_root
                        .join("out")
                        .join(format!("{}_world.json", world_label));
                    if world_path != canonical_world {
                        if let Some(parent) = canonical_world.parent() {
                            let _ = fs::create_dir_all(parent);
                        }
                        if let Err(e) = fs::copy(&world_path, &canonical_world) {
                            let _ =
                                tx.send(format!("__DONE_ERR__copy world for map failed: {}", e));
                            return;
                        }
                    }
                    let map_script = repo_root.join("generate_canonical_map.py");
                    if map_script.exists() {
                        send("(builder) running generate_canonical_map ...".into(), &tx);
                        let mut cmd = Command::new("python");
                        cmd.current_dir(&repo_root);
                        cmd.arg(&map_script);
                        cmd.arg("--games");
                        cmd.arg(&world_label);
                        cmd.arg("--world-dir");
                        cmd.arg(repo_root.join("out"));
                        cmd.arg("--out-dir");
                        cmd.arg(repo_root.join("assets/maps"));
                        if let Err(e) = run_command_stream_channel(
                            &mut cmd,
                            "generate_canonical_map",
                            tx.clone(),
                        ) {
                            let _ = tx.send(format!("__DONE_ERR__{}", e));
                            return;
                        }
                        send("(builder) map regeneration completed".into(), &tx);
                    } else {
                        send(
                            format!(
                                "(builder) generate_canonical_map.py not found at {}",
                                map_script.display()
                            ),
                            &tx,
                        );
                    }
                    let _ = tx.send("__DONE_OK__".into());
                });
            }
            "ctx" => {
                let room = self.ctx.room.as_deref().unwrap_or("<none>");
                let inv = if self.ctx.inventory.is_empty() {
                    "<empty>".to_string()
                } else {
                    self.ctx.inventory.join(", ")
                };
                self.push_plain(format!("(context) room={} inventory={}", room, inv));
            }
            _ => {
                self.push_plain("(context) unknown meta-command");
            }
        }
    }

    fn handle_world_events(&mut self, ctx: &egui::Context) {
        if let Some(parser) = self.parser.as_ref() {
            if let Some(world) = parser.world.as_ref() {
                let events = process_world_events(&mut self.ctx, world, &mut self.engine);
                for msg in events.messages {
                    self.push_plain(self.prettify_sentence(&msg));
                }
                if events.room_changed {
                    self.reload_room_media(ctx);
                    self.log_room_description(true, false, true);
                }
            }
        }
        if !self.engine.notifications.is_empty() {
            let notes: Vec<String> = self.engine.notifications.drain(..).collect();
            for note in notes {
                let pretty = self.prettify_sentence(&note);
                self.push_plain(pretty.clone());
                self.record_status(StatusLine {
                    message: pretty,
                    success: true,
                });
            }
        }
    }

    fn record_status(&mut self, status: StatusLine) {
        if self.show_system_messages && !status.message.starts_with("(move") {
            self.push_plain(status.message.clone());
        }
        self.status = Some(status);
        self.status_time = Some(self.current_time);
    }

    fn speech_elapsed_tag(&self) -> String {
        self.speech_started_at
            .map(|t| format!(" [t={:.2}s]", t.elapsed().as_secs_f32()))
            .unwrap_or_default()
    }

    fn poll_speech_queue(&mut self, ctx: &egui::Context) {
        let Some(rx) = self.speech_rx.as_ref() else {
            return;
        };
        let mut events = Vec::new();
        loop {
            match rx.try_recv() {
                Ok(ev) => events.push(ev),
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    self.speech_tx = None;
                    self.speech_rx = None;
                    self.speech_state = SpeechUiState::Idle;
                    self.record_status(StatusLine {
                        message: "(speech) Disconnected from capture thread".into(),
                        success: false,
                    });
                    return;
                }
            }
        }
        for ev in events {
            match ev {
                SpeechEvent::RecordingStarted => {
                    self.speech_state = SpeechUiState::Recording;
                    self.speech_started_at = Some(Instant::now());
                    self.record_status(StatusLine {
                        message: format!("(speech) Recording…{}", self.speech_elapsed_tag()),
                        success: true,
                    });
                }
                SpeechEvent::RecordingStopped => {
                    self.speech_state = SpeechUiState::Idle;
                    self.record_status(StatusLine {
                        message: format!("(speech) Stopped recording{}", self.speech_elapsed_tag()),
                        success: true,
                    });
                }
                SpeechEvent::Processing => {
                    self.speech_state = SpeechUiState::Processing;
                    self.record_status(StatusLine {
                        message: format!("(speech) Processing…{}", self.speech_elapsed_tag()),
                        success: true,
                    });
                }
                SpeechEvent::Info(msg) => {
                    self.record_status(StatusLine {
                        message: format!("(speech) {}{}", msg, self.speech_elapsed_tag()),
                        success: true,
                    });
                }
                SpeechEvent::Transcript(text) => {
                    self.speech_state = SpeechUiState::Idle;
                    let elapsed = self
                        .speech_started_at
                        .map(|t| t.elapsed().as_secs_f32())
                        .unwrap_or(0.0);
                    if !text.trim().is_empty() {
                        self.input = text;
                        println!(
                            "[speech] transcript (t={:.2}s): {}",
                            elapsed,
                            self.input.trim()
                        );
                        if self.speech_auto_send {
                            self.handle_input(ctx);
                        } else {
                            self.input_has_focus = true;
                            self.record_status(StatusLine {
                                message: format!(
                                    "(speech) Transcript captured (press Enter to send) [t={:.2}s]",
                                    elapsed
                                ),
                                success: true,
                            });
                        }
                    } else {
                        self.record_status(StatusLine {
                            message: format!(
                                "(speech) Empty transcript{}",
                                self.speech_elapsed_tag()
                            ),
                            success: false,
                        });
                    }
                    self.speech_started_at = None;
                }
                SpeechEvent::Error(msg) => {
                    self.speech_state = SpeechUiState::Idle;
                    self.record_status(StatusLine {
                        message: format!("(speech) {}{}", msg, self.speech_elapsed_tag()),
                        success: false,
                    });
                    self.speech_started_at = None;
                }
            }
        }
    }

    fn ensure_speech_runtime(&mut self) -> Result<(), String> {
        if self.speech_tx.is_some() && self.speech_rx.is_some() {
            return Ok(());
        }
        let model_path = self.speech_model_path.clone();
        let language = self.speech_language.clone();
        match spawn_speech_runtime(&model_path, language) {
            Ok((tx, rx)) => {
                self.speech_tx = Some(tx);
                self.speech_rx = Some(rx);
                Ok(())
            }
            Err(e) => Err(e),
        }
    }

    fn start_speech_capture(&mut self) {
        if !self.speech_enabled {
            return;
        }
        if !SPEECH_BUILD_ENABLED {
            self.record_status(StatusLine {
                message: "(speech) Not available in this build (enable feature \"speech_input\")"
                    .into(),
                success: false,
            });
            return;
        }
        if let Err(e) = self.ensure_speech_runtime() {
            self.record_status(StatusLine {
                message: format!("(speech) {}", e),
                success: false,
            });
            return;
        }
        if let Some(tx) = self.speech_tx.as_ref() {
            let _ = tx.send(SpeechCommand::Start {
                model_path: self.speech_model_path.clone(),
                language: self.speech_language.clone(),
            });
        }
    }

    fn stop_speech_capture(&mut self) {
        if let Some(tx) = self.speech_tx.as_ref() {
            let _ = tx.send(SpeechCommand::Stop);
        }
    }

    fn pulse_window(&mut self, text: &str, now: f64) -> Option<f64> {
        if let Some(t) = &self.pulse_text {
            if t != text {
                return None;
            }
        } else {
            return None;
        }
        if now >= self.pulse_next && now <= self.pulse_next + self.pulse_duration {
            return Some(self.pulse_next);
        }
        if now > self.pulse_next + self.pulse_duration {
            self.pulse_next = now + self.pulse_interval;
        }
        None
    }

    fn refresh_pulse(&mut self, now: f64) {
        let Some(start) = self.pulse_state(now) else {
            return;
        };
        let pulse_text = match self.pulse_text.clone() {
            Some(t) => t,
            None => return,
        };
        if self.pulse_spans.is_empty() {
            return;
        }
        let pal = self.current_palette();
        let (mut normal_fmt, mut emphasis_fmt) = self.text_formats();
        normal_fmt.color = pal.fg_contrast;
        emphasis_fmt.color = pal.fg_contrast;
        let job = highlight_items(
            &pulse_text,
            &[],
            normal_fmt.clone(),
            emphasis_fmt.clone(),
            self.pulse_spans.clone(),
            Some(start),
            now,
            self.pulse_duration,
        );
        if let Some(idx) = self.log.iter().position(|e| match e {
            LogEntry::Styled { plain, .. } => plain == &pulse_text,
            _ => false,
        }) {
            self.log[idx] = LogEntry::Styled {
                job,
                plain: pulse_text,
            };
        }
    }

    fn schedule_pulse_repaint(&mut self, ctx: &egui::Context, now: f64) {
        if self.pulse_text.is_none() || self.pulse_spans.is_empty() {
            return;
        }
        let dur = self.pulse_duration.max(0.1);
        if now < self.pulse_next {
            let wait = (self.pulse_next - now).max(0.01);
            ctx.request_repaint_after(std::time::Duration::from_secs_f64(wait));
        } else if now <= self.pulse_next + dur {
            ctx.request_repaint_after(std::time::Duration::from_millis(33));
        } else {
            self.pulse_next = now + self.pulse_interval.max(1.0);
            ctx.request_repaint_after(std::time::Duration::from_secs_f64(
                (self.pulse_next - now).max(0.01),
            ));
        }
    }

    fn pulse_state(&mut self, now: f64) -> Option<f64> {
        if self.pulse_text.is_none() || self.pulse_spans.is_empty() {
            return None;
        }
        if now >= self.pulse_next && now <= self.pulse_next + self.pulse_duration {
            return Some(self.pulse_next);
        }
        if now > self.pulse_next + self.pulse_duration {
            self.pulse_next = now + self.pulse_interval;
        }
        None
    }

    fn resolve_object_id(&self, token: &str) -> Option<String> {
        let upper = token.to_uppercase();
        let parser = self.parser.as_ref()?;
        let world = parser.world.as_ref()?;
        if world.objects.contains_key(&upper) {
            return Some(upper);
        }
        for (id, obj) in world.objects.iter() {
            if obj.names.iter().any(|n| n.eq_ignore_ascii_case(token)) {
                return Some(id.to_uppercase());
            }
        }
        None
    }

    fn object_in_scope(&self, obj_upper: &str) -> bool {
        if self
            .ctx
            .inventory
            .iter()
            .any(|i| i.eq_ignore_ascii_case(obj_upper))
        {
            return true;
        }
        if let (Some(idx), Some(world), Some(room)) = (
            self.parser.as_ref().and_then(|p| p.world_index.as_ref()),
            self.parser.as_ref().and_then(|p| p.world.as_ref()),
            self.ctx.room.as_ref(),
        ) {
            let visibles = objects_visible_in_room(world, idx, &room.to_uppercase());
            return visibles.iter().any(|o| o.eq_ignore_ascii_case(obj_upper));
        }
        false
    }

    fn parse_effect_tokens(raw: &str) -> ParsedEffects {
        let mut out = ParsedEffects::default();
        for token in raw.split(|c: char| c == ',' || c.is_whitespace()) {
            let t = token.trim();
            if t.is_empty() {
                continue;
            }
            let up = t.to_uppercase();
            if let Some((val, absolute)) = parse_credits_change(&up) {
                out.credits = Some((val, absolute));
                continue;
            }
            if let Some(rest) = up.strip_prefix("GIVE:").or_else(|| up.strip_prefix("G:")) {
                let id = rest.trim();
                if !id.is_empty() {
                    out.give_items.push(normalize_id(id));
                }
                continue;
            }
            if let Some(rest) = up.strip_prefix("TAKE:").or_else(|| up.strip_prefix("T:")) {
                let id = rest.trim();
                if !id.is_empty() {
                    out.take_items.push(normalize_id(id));
                }
                continue;
            }
            if up.contains("+=") || up.contains("-=") || up.contains('=') {
                out.counter_changes.push(up);
                continue;
            }
            if up.starts_with('!') || up.starts_with('-') {
                out.clear_flags
                    .push(up.trim_start_matches(|c| c == '!' || c == '-').to_string());
            } else {
                out.set_flags.push(up);
            }
        }
        out
    }

    fn log_dialog_view(&mut self, view: &DialogView) {
        self.push_plain(format!("(npc:{}) {}", view.npc_id, view.text));
        if view.options.is_empty() {
            self.push_plain("(npc) [dialog ended]");
        } else {
            for (i, opt) in view.options.iter().enumerate() {
                self.push_plain(format!("[{}] {}", i + 1, opt.prompt));
            }
        }
    }

    fn log_room_description(&mut self, force_long: bool, force_tts: bool, show_move_header: bool) {
        let Some(parser) = self.parser.as_ref() else {
            return;
        };
        let Some(world) = parser.world.as_ref() else {
            return;
        };
        let opts = DescribeOptions {
            force_long,
            include_visible: self.show_hints,
            include_exits: true,
            prefer_enhanced: true,
        };
        if let Some(desc) = describe_room(
            &self.ctx,
            world,
            parser.world_index.as_ref(),
            Some(&mut self.engine),
            &opts,
        ) {
            // Update fogged map state based on exits actually shown
            let room_name = world
                .rooms
                .get(&desc.room_id)
                .and_then(|r| r.desc.as_deref());
            self.engine.map_state.update_from_exits(
                &desc.room_id,
                room_name,
                &desc.exits,
                parser.canonical_map.as_ref(),
                self.engine.tick,
            );

            let visible_pretty: Vec<String> = desc
                .visible_items
                .iter()
                .map(|id| self.prettify_id(id))
                .collect();
            let npc_pretty: Vec<String> = desc
                .visible_npcs
                .iter()
                .map(|id| self.prettify_id(id))
                .collect();
            let exits_pretty: Vec<String> =
                desc.exits.iter().map(|e| self.prettify_exit(e)).collect();

            let pretty_room = self.prettify_id(&desc.room_id);
            if show_move_header {
                if let Some(dir) = self.last_move_dir.take() {
                    self.push_plain(format!("(move {}) moved to {}", dir, pretty_room));
                } else {
                    self.push_plain(format!("(move) moved to {}", pretty_room));
                }
                self.push_plain(String::new()); // intentional blank line
            }

            if desc.first_visit {
                let pal = self.current_palette();
                let (mut normal_fmt, _) = self.text_formats();
                normal_fmt.color = pal.fg_contrast;
                let mut italics = normal_fmt.clone();
                italics.italics = true;
                let mut job = LayoutJob::default();
                job.append("(first visit)", 0.0, italics);
                self.push_styled("(first visit)".to_string(), job);
            }
            let pretty_text = self.prettify_sentence(&desc.text);
            let empty: Vec<String> = Vec::new();
            let show_hints_line = self.show_hints && self.show_system_messages;
            let items_for_highlight: &[String] = if self.show_hints {
                &visible_pretty
            } else {
                &empty
            };
            let items_for_hints: &[String] = if show_hints_line {
                &visible_pretty
            } else {
                &empty
            };
            let pal = self.current_palette();
            let (mut normal_fmt, mut emphasis_fmt) = self.text_formats();
            normal_fmt.color = pal.fg_contrast;
            emphasis_fmt.color = pal.fg_contrast;
            let mut job = highlight_items(
                &pretty_text,
                items_for_highlight,
                normal_fmt.clone(),
                emphasis_fmt.clone(),
                highlight_spans(&pretty_text, items_for_highlight),
                self.pulse_window(&pretty_text, self.current_time),
                self.current_time,
                self.pulse_duration,
            );
            for sec in job.sections.iter_mut() {
                sec.format.color = pal.fg_contrast;
            }
            self.push_styled(pretty_text.clone(), job);
            for line in desc.object_lines.iter() {
                let pretty_line = self.prettify_sentence(line);
                let mut obj_job = highlight_items(
                    &pretty_line,
                    items_for_highlight,
                    normal_fmt.clone(),
                    emphasis_fmt.clone(),
                    highlight_spans(&pretty_line, items_for_highlight),
                    self.pulse_window(&pretty_line, self.current_time),
                    self.current_time,
                    self.pulse_duration,
                );
                for sec in obj_job.sections.iter_mut() {
                    sec.format.color = pal.fg_contrast;
                }
                self.push_styled(pretty_line.clone(), obj_job);
            }
            if !items_for_hints.is_empty() {
                self.push_plain(format!("(hint) You see: {}", items_for_hints.join(", ")));
            }
            if !npc_pretty.is_empty() {
                self.push_plain(format!(
                    "{} {} here.",
                    npc_pretty.join(", "),
                    if npc_pretty.len() > 1 { "are" } else { "is" }
                ));
            }
            self.push_plain(String::new()); // intentional blank line
            if !exits_pretty.is_empty() {
                self.push_plain(format!("Exits: {}", exits_pretty.join(", ")));
            }
            self.maybe_play_room_tts(&desc.room_id, desc.first_visit, force_tts);
            // Track spans for potential future pulses
            self.pulse_spans = highlight_spans(&pretty_text, items_for_highlight);
            self.pulse_text = Some(pretty_text.clone());
            self.pulse_next = self.current_time + 0.6; // first pulse soon after render
        }
    }

    fn should_play_tts(&self, room_id: &str, first_visit: bool, force: bool) -> bool {
        match self.tts_mode {
            TtsMode::Never => false,
            TtsMode::Always => true,
            TtsMode::FirstTime => {
                let id = room_id.to_uppercase();
                if force {
                    return true;
                }
                if !first_visit && self.tts_played_rooms.contains(&id) {
                    return false;
                }
                !self.tts_played_rooms.contains(&id)
            }
        }
    }

    fn maybe_play_room_tts(&mut self, room_id: &str, first_visit: bool, force: bool) {
        if !self.should_play_tts(room_id, first_visit, force) {
            return;
        }
        let manifest = match self.manifest.as_ref() {
            Some(m) => m,
            None => return,
        };
        let handle = match self.tts_handle.as_ref() {
            Some(h) => h.clone(),
            None => return,
        };
        let base_assets = self.assets_base.as_path();
        let Some(path) = room_tts_path(manifest, room_id, base_assets) else {
            return;
        };
        if !path.exists() {
            self.record_status(StatusLine {
                message: format!("(tts) missing {}", path.display()),
                success: false,
            });
            return;
        }

        // Cancel any in-flight playback immediately.
        if let Some(sink) = self.tts_sink.take() {
            sink.stop();
        }
        if let Some(cancel) = self.tts_cancel.take() {
            cancel.store(true, Ordering::Relaxed);
        }

        let cancel_flag = Arc::new(AtomicBool::new(false));
        self.tts_cancel = Some(cancel_flag.clone());

        let vol = self.tts_volume.clamp(0.0, 1.0);
        let path_buf = path.clone();
        let sink = match Sink::try_new(&handle) {
            Ok(s) => s,
            Err(_) => return,
        };
        let sink_arc = Arc::new(sink);
        self.tts_sink = Some(sink_arc.clone());
        thread::spawn(move || {
            // Debounce to avoid stutter during rapid traversal
            thread::sleep(Duration::from_millis(150));
            if cancel_flag.load(Ordering::Relaxed) {
                sink_arc.stop();
                return;
            }
            let file = match std::fs::File::open(&path_buf) {
                Ok(f) => f,
                Err(_) => return,
            };
            let decoder = match Decoder::new(BufReader::new(file)) {
                Ok(d) => d,
                Err(_) => return,
            };
            sink_arc.set_volume(vol);
            sink_arc.append(decoder);
            while !sink_arc.empty() {
                if cancel_flag.load(Ordering::Relaxed) {
                    sink_arc.stop();
                    break;
                }
                thread::sleep(Duration::from_millis(25));
            }
        });
        self.tts_played_rooms.insert(room_id.to_uppercase());
    }

    fn log_object_description(&mut self, obj_id: &str) {
        if let Some(parser) = self.parser.as_ref() {
            if let Some(world) = parser.world.as_ref() {
                if let Some(desc) = describe_object(obj_id, world, true) {
                    let pretty = self.prettify_id(obj_id);
                    let (normal_fmt, emphasis_fmt) = self.text_formats();
                    let job = highlight_items(
                        &desc.text,
                        &[pretty.clone()],
                        normal_fmt.clone(),
                        emphasis_fmt.clone(),
                        highlight_spans(&desc.text, &[pretty.clone()]),
                        None,
                        self.current_time,
                        self.pulse_duration,
                    );
                    self.push_styled(desc.text.clone(), job);
                    return;
                }
            }
        }
        self.push_plain(format!("You see nothing special about {}.", obj_id));
    }

    fn handle_look_intent(&mut self, intent: &ScoredIntent) -> bool {
        let is_look = intent
            .verb
            .as_ref()
            .map(|v| v.token.eq_ignore_ascii_case("LOOK"))
            .unwrap_or(false);
        if !is_look {
            return false;
        }
        if let Some(noun) = intent.noun.as_ref() {
            if let Some(obj_id) = self.resolve_object_id(&noun.token) {
                if self.object_in_scope(&obj_id) {
                    self.log_object_description(&obj_id);
                } else {
                    self.push_plain(format!("You don't see {} here.", obj_id));
                }
                return true;
            }
        }
        self.log_room_description(true, true, false);
        true
    }

    fn handle_action_intent(&mut self, intent: &ScoredIntent) -> bool {
        self.handle_talk_intent(intent)
            || self.handle_object_action_intent(intent)
            || self.handle_wait_intent(intent)
            || self.handle_take_intent(intent)
            || self.handle_drop_put_intent(intent)
            || self.handle_open_close_intent(intent)
            || self.handle_lock_unlock_intent(intent)
            || self.handle_read_intent(intent)
    }

    fn handle_talk_intent(&mut self, intent: &ScoredIntent) -> bool {
        let Some(verb) = intent.verb.as_ref() else {
            return false;
        };
        if !verb.token.eq_ignore_ascii_case("TALK") {
            return false;
        }
        let Some(room) = self.ctx.room.clone() else {
            self.push_plain("(dialog) no current room");
            return true;
        };
        let Some(parser) = self.parser.as_mut() else {
            self.push_plain("(dialog) no parser loaded");
            return true;
        };
        let Some(world) = parser.world.as_ref() else {
            self.push_plain("(dialog) no world loaded; use /newworld first");
            return true;
        };
        let npcs_here: Vec<(String, &WorldNpc)> = world
            .npcs
            .iter()
            .filter(|(_, npc)| {
                npc.location
                    .as_ref()
                    .map(|l| l.eq_ignore_ascii_case(&room))
                    .unwrap_or(false)
            })
            .map(|(id, npc)| (id.clone(), npc))
            .collect();
        if npcs_here.is_empty() {
            self.push_plain("No one here to talk to.");
            return true;
        }
        let mut target: Option<String> = None;
        if let Some(noun) = intent.noun.as_ref() {
            let norm = normalize_id(&noun.token);
            for (id, npc) in npcs_here.iter() {
                if id.eq_ignore_ascii_case(&norm)
                    || npc
                        .name
                        .as_ref()
                        .map(|n| n.eq_ignore_ascii_case(&noun.token))
                        .unwrap_or(false)
                {
                    target = Some(id.clone());
                    break;
                }
            }
        }
        if target.is_none() && npcs_here.len() == 1 {
            target = Some(npcs_here[0].0.clone());
        }
        if target.is_none() {
            let names: Vec<String> = npcs_here
                .iter()
                .map(|(id, npc)| npc.name.clone().unwrap_or_else(|| id.clone()))
                .collect();
            self.push_plain(format!(
                "Who do you want to talk to? Available: {}",
                names.join(", ")
            ));
            return true;
        }
        let npc_id = target.unwrap();
        self.start_dialog_with(&npc_id, None);
        true
    }

    fn handle_object_action_intent(&mut self, intent: &ScoredIntent) -> bool {
        let Some(noun) = intent.noun.as_ref() else {
            return false;
        };
        let Some(verb) = intent.verb.as_ref() else {
            return false;
        };
        let Some(parser) = self.parser.as_mut() else {
            return false;
        };
        if let (Some(world), Some(idx)) = (parser.world.as_mut(), parser.world_index.as_mut()) {
            if let Some(result) = apply_object_action(
                world,
                idx,
                &mut self.ctx,
                &mut self.engine,
                &verb.token,
                &noun.token,
            ) {
                match result {
                    Ok(msg) => {
                        let pretty = self.prettify_sentence(&msg);
                        self.record_status(StatusLine {
                            message: pretty.clone(),
                            success: true,
                        });
                        self.push_plain(pretty);
                        self.log_room_description(true, false, false);
                    }
                    Err(err) => {
                        let pretty = self.prettify_sentence(&err);
                        self.record_status(StatusLine {
                            message: pretty.clone(),
                            success: false,
                        });
                        self.push_plain(pretty);
                    }
                }
                return true;
            }
        }
        false
    }

    fn handle_wait_intent(&mut self, intent: &ScoredIntent) -> bool {
        let is_wait = intent
            .verb
            .as_ref()
            .map(|v| v.token.eq_ignore_ascii_case("WAIT"))
            .unwrap_or(false);
        if !is_wait {
            return false;
        }
        self.engine.step();
        self.push_plain("Time passes...");
        true
    }

    fn handle_take_intent(&mut self, intent: &ScoredIntent) -> bool {
        let is_take = intent
            .verb
            .as_ref()
            .map(|v| {
                matches!(
                    v.token.to_uppercase().as_str(),
                    "TAKE" | "GET" | "GRAB" | "PICK"
                )
            })
            .unwrap_or(false);
        if !is_take {
            return false;
        }
        let Some(noun) = intent.noun.as_ref() else {
            self.push_plain("Take what?");
            return true;
        };
        if let Some(parser) = self.parser.as_mut() {
            if let (Some(world), Some(idx)) = (parser.world.as_mut(), parser.world_index.as_mut()) {
                match zork_core::take_object(world, idx, &mut self.ctx, &noun.token) {
                    Ok(msg) => {
                        let pretty = self.prettify_sentence(&msg);
                        self.push_plain(pretty.clone());
                        self.record_status(StatusLine {
                            message: pretty,
                            success: true,
                        });
                        self.log_room_description(true, false, false);
                    }
                    Err(e) => {
                        let pretty = self.prettify_sentence(&e);
                        self.push_plain(pretty.clone());
                        self.record_status(StatusLine {
                            message: pretty,
                            success: false,
                        });
                    }
                }
            }
        }
        true
    }

    fn handle_drop_put_intent(&mut self, intent: &ScoredIntent) -> bool {
        let verb = intent
            .verb
            .as_ref()
            .map(|v| v.token.to_uppercase())
            .unwrap_or_default();
        let is_put = verb == "PUT";
        let is_drop = verb == "DROP" || verb == "LEAVE";
        if !is_put && !is_drop {
            return false;
        }
        let Some(noun) = intent.noun.as_ref() else {
            self.push_plain(format!("{} what?", verb.to_lowercase()));
            return true;
        };
        if let Some(parser) = self.parser.as_mut() {
            if let (Some(world), Some(idx)) = (parser.world.as_mut(), parser.world_index.as_mut()) {
                if is_put {
                    if let Some(io) = intent.indirect.as_ref() {
                        match zork_core::put_object_in(
                            world,
                            idx,
                            &mut self.ctx,
                            &noun.token,
                            &io.token,
                        ) {
                            Ok(msg) => {
                                let pretty = self.prettify_sentence(&msg);
                                self.push_plain(pretty.clone());
                                self.record_status(StatusLine {
                                    message: pretty,
                                    success: true,
                                });
                                self.log_room_description(true, false, false);
                            }
                            Err(e) => {
                                let pretty = self.prettify_sentence(&e);
                                self.push_plain(pretty.clone());
                                self.record_status(StatusLine {
                                    message: pretty,
                                    success: false,
                                });
                            }
                        }
                        return true;
                    } else {
                        self.push_plain("Put it where?");
                        return true;
                    }
                }
                match zork_core::drop_object(world, idx, &mut self.ctx, &noun.token) {
                    Ok(msg) => {
                        let pretty = self.prettify_sentence(&msg);
                        self.push_plain(pretty.clone());
                        self.record_status(StatusLine {
                            message: pretty,
                            success: true,
                        });
                        self.log_room_description(true, false, false);
                    }
                    Err(e) => {
                        let pretty = self.prettify_sentence(&e);
                        self.push_plain(pretty.clone());
                        self.record_status(StatusLine {
                            message: pretty,
                            success: false,
                        });
                    }
                }
            }
        }
        true
    }

    fn handle_open_close_intent(&mut self, intent: &ScoredIntent) -> bool {
        let verb = intent
            .verb
            .as_ref()
            .map(|v| v.token.to_uppercase())
            .unwrap_or_default();
        let (is_open, is_close) = (verb == "OPEN", verb == "CLOSE");
        if !is_open && !is_close {
            return false;
        }
        let Some(noun) = intent.noun.as_ref() else {
            self.push_plain(format!("{} what?", verb.to_lowercase()));
            return true;
        };
        if let Some(parser) = self.parser.as_mut() {
            if let (Some(world), Some(idx)) = (parser.world.as_mut(), parser.world_index.as_mut()) {
                let ctx = self.ctx.clone();
                let engine = &mut self.engine;
                match zork_core::open_close_object(
                    world,
                    idx,
                    &ctx,
                    Some(engine),
                    &noun.token,
                    is_open,
                ) {
                    Ok(msg) => {
                        let pretty = self.prettify_sentence(&msg);
                        self.push_plain(pretty.clone());
                        self.record_status(StatusLine {
                            message: pretty,
                            success: true,
                        });
                        self.log_room_description(true, false, false);
                    }
                    Err(e) => {
                        let pretty = self.prettify_sentence(&e);
                        self.push_plain(pretty.clone());
                        self.record_status(StatusLine {
                            message: pretty,
                            success: false,
                        });
                    }
                }
            }
        }
        true
    }

    fn handle_lock_unlock_intent(&mut self, intent: &ScoredIntent) -> bool {
        let verb = intent
            .verb
            .as_ref()
            .map(|v| v.token.to_uppercase())
            .unwrap_or_default();
        let (is_lock, is_unlock) = (verb == "LOCK", verb == "UNLOCK");
        if !is_lock && !is_unlock {
            return false;
        }
        let Some(noun) = intent.noun.as_ref() else {
            self.push_plain(format!("{} what?", verb.to_lowercase()));
            return true;
        };
        if let Some(parser) = self.parser.as_mut() {
            if let (Some(world), Some(idx)) = (parser.world.as_mut(), parser.world_index.as_mut()) {
                let ctx = self.ctx.clone();
                match zork_core::lock_unlock_object(world, idx, &ctx, &noun.token, is_lock) {
                    Ok(msg) => {
                        let pretty = self.prettify_sentence(&msg);
                        self.push_plain(pretty.clone());
                        self.record_status(StatusLine {
                            message: pretty,
                            success: true,
                        });
                        self.log_room_description(true, false, false);
                    }
                    Err(e) => {
                        let pretty = self.prettify_sentence(&e);
                        self.push_plain(pretty.clone());
                        self.record_status(StatusLine {
                            message: pretty,
                            success: false,
                        });
                    }
                }
            }
        }
        true
    }

    fn handle_read_intent(&mut self, intent: &ScoredIntent) -> bool {
        let verb = intent
            .verb
            .as_ref()
            .map(|v| v.token.to_uppercase())
            .unwrap_or_default();
        if verb != "READ" {
            return false;
        }
        let Some(noun) = intent.noun.as_ref() else {
            self.push_plain("Read what?");
            return true;
        };
        if let Some(parser) = self.parser.as_mut() {
            if let (Some(world), Some(idx)) = (parser.world.as_mut(), parser.world_index.as_mut()) {
                match zork_core::read_object(world, idx, &self.ctx, &noun.token) {
                    Ok(msg) => {
                        let pretty = self.prettify_sentence(&msg);
                        self.push_plain(pretty.clone());
                        self.record_status(StatusLine {
                            message: pretty,
                            success: true,
                        });
                    }
                    Err(e) => {
                        let pretty = self.prettify_sentence(&e);
                        self.push_plain(pretty.clone());
                        self.record_status(StatusLine {
                            message: pretty,
                            success: false,
                        });
                    }
                }
            }
        }
        true
    }
}

fn highlight_items(
    text: &str,
    items: &[String],
    normal: TextFormat,
    emphasis: TextFormat,
    pulse_spans: Vec<(usize, usize)>,
    pulse_start: Option<f64>,
    pulse_now: f64,
    pulse_duration: f64,
) -> LayoutJob {
    let mut job = LayoutJob::default();
    if text.is_empty() {
        job.append(text, 0.0, normal);
        return job;
    }
    let pulse_progress = pulse_start
        .filter(|ps| pulse_active(pulse_now, *ps, pulse_duration))
        .map(|ps| ((pulse_now - ps) / pulse_duration).clamp(0.0, 1.0));
    let spans = if pulse_progress.is_some() && !pulse_spans.is_empty() {
        pulse_spans
    } else if !items.is_empty() {
        highlight_spans(text, items)
    } else {
        job.append(text, 0.0, normal);
        return job;
    };
    let mut cursor = 0;
    for (s, e) in spans.iter() {
        if *s > cursor {
            job.append(&text[cursor..*s], 0.0, normal.clone());
        }
        if *e <= text.len() {
            if let Some(progress) = pulse_progress {
                append_pulsed_span(&mut job, &text[*s..*e], emphasis.clone(), progress);
            } else {
                job.append(&text[*s..*e], 0.0, emphasis.clone());
            }
        }
        cursor = *e;
    }
    if cursor < text.len() {
        job.append(&text[cursor..], 0.0, normal);
    }
    job
}

fn pulse_active(now: f64, start: f64, dur: f64) -> bool {
    dur > 0.0 && now >= start && now <= start + dur
}

fn append_pulsed_span(job: &mut LayoutJob, span_text: &str, emphasis: TextFormat, progress: f64) {
    let chars: Vec<(usize, char)> = span_text.char_indices().collect();
    if chars.is_empty() {
        job.append(span_text, 0.0, emphasis);
        return;
    }
    let count = chars.len();
    // Sweep starts just before the first letter and ends just after the last.
    let center = (progress.clamp(0.0, 1.0) * (count as f64 + 2.0)) - 1.0;
    let width = 1.75_f64; // looser falloff for a softer glow across neighbors
    let amplitude = 2.0_f64; // peak brightness ~2.4x at the wave crest for visibility
    for (i, (start, _ch)) in chars.iter().enumerate() {
        let start = *start;
        let end = if i + 1 < count {
            chars[i + 1].0
        } else {
            span_text.len()
        };
        let distance = (i as f64 - center).abs();
        let falloff = (1.0 - distance / width).clamp(0.0, 1.0);
        let brightness = 1.0 + amplitude * falloff;
        let mut fmt = emphasis.clone();
        fmt.color = fmt.color.gamma_multiply(brightness as f32);
        job.append(&span_text[start..end], 0.0, fmt);
    }
}

fn highlight_spans(text: &str, items: &[String]) -> Vec<(usize, usize)> {
    let mut ranges: Vec<(usize, usize)> = Vec::new();
    for it in items {
        let token = it.trim();
        if token.is_empty() {
            continue;
        }
        for (start, end) in find_word_boundaries(text, token) {
            ranges.push((start, end));
        }
    }
    ranges.sort_by_key(|r| r.0);
    let mut merged: Vec<(usize, usize)> = Vec::new();
    for (s, e) in ranges {
        if let Some((_, me)) = merged.last_mut() {
            if e <= *me {
                continue;
            }
            if s <= *me {
                *me = e;
                continue;
            }
        }
        merged.push((s, e));
    }
    merged
}

fn split_word(s: &str) -> (&str, &str, &str) {
    let mut start_idx: Option<usize> = None;
    let mut end_idx: Option<usize> = None;
    for (i, ch) in s.char_indices() {
        if ch.is_alphanumeric() || ch == '-' {
            start_idx = Some(i);
            break;
        }
    }
    for (i, ch) in s.char_indices().rev() {
        if ch.is_alphanumeric() || ch == '-' {
            end_idx = Some(i + ch.len_utf8());
            break;
        }
    }
    let (start, end) = match (start_idx, end_idx) {
        (Some(sidx), Some(eidx)) if sidx < eidx => (sidx, eidx),
        _ => (0, 0),
    };
    let prefix = &s[..start];
    let core = &s[start..end];
    let suffix = &s[end..];
    (prefix, core, suffix)
}

fn find_word_boundaries(haystack: &str, needle: &str) -> Vec<(usize, usize)> {
    let mut ranges = Vec::new();
    if haystack.is_empty() || needle.is_empty() {
        return ranges;
    }
    let hay_lower = haystack.to_lowercase();
    let needle_lower = needle.to_lowercase();
    let mut search_start = 0;
    while let Some(idx) = hay_lower[search_start..].find(&needle_lower) {
        let start = search_start + idx;
        let end = start + needle_lower.len();
        let before = hay_lower[..start].chars().last();
        let after = hay_lower[end..].chars().next();
        let is_boundary_before = before.map_or(true, |c| !c.is_alphanumeric() && c != '-');
        let is_boundary_after = after.map_or(true, |c| !c.is_alphanumeric() && c != '-');
        if is_boundary_before && is_boundary_after {
            ranges.push((start, end));
        }
        search_start = end;
    }
    ranges
}

fn build_pano_mesh(
    rect: egui::Rect,
    texture_id: egui::TextureId,
    yaw_deg: f32,
    pitch_deg: f32,
    fov_y_deg: f32,
) -> egui::Mesh {
    let cols: usize = 48;
    let rows: usize = 24;
    let yaw = yaw_deg.to_radians();
    let pitch = pitch_deg
        .to_radians()
        .clamp(-MAX_PITCH_DEG.to_radians(), MAX_PITCH_DEG.to_radians());
    let fov_y = fov_y_deg
        .to_radians()
        .clamp(0.2, std::f32::consts::PI - 0.2);
    let aspect = (rect.width().max(1.0)) / (rect.height().max(1.0));
    let fov_x = 2.0 * ((fov_y * 0.5).tan() * aspect).atan();

    let mut mesh = egui::Mesh::with_texture(texture_id);
    mesh.vertices.reserve((cols + 1) * (rows + 1));
    mesh.indices.reserve(cols * rows * 6);

    for y in 0..=rows {
        let v = y as f32 / rows as f32;
        let screen_y = egui::lerp(rect.top()..=rect.bottom(), v);
        let theta = (1.0 - 2.0 * v) * fov_y * 0.5;
        for x in 0..=cols {
            let u = x as f32 / cols as f32;
            let screen_x = egui::lerp(rect.left()..=rect.right(), u);
            let phi = (2.0 * u - 1.0) * fov_x * 0.5;

            let lon = yaw + phi;
            let lat = (pitch + theta).clamp(
                -std::f32::consts::FRAC_PI_2 + 0.001,
                std::f32::consts::FRAC_PI_2 - 0.001,
            );

            let tex_u = (lon / (2.0 * std::f32::consts::PI)).rem_euclid(1.0);
            let tex_v = 0.5 - (lat / std::f32::consts::PI);

            mesh.vertices.push(egui::epaint::Vertex {
                pos: egui::pos2(screen_x, screen_y),
                uv: egui::pos2(tex_u, tex_v),
                color: egui::Color32::WHITE,
            });
        }
    }

    for y in 0..rows {
        for x in 0..cols {
            let i0 = (y * (cols + 1) + x) as u32;
            let i1 = i0 + 1;
            let i2 = i0 + (cols + 1) as u32;
            let i3 = i2 + 1;
            mesh.indices.extend_from_slice(&[i0, i2, i1, i1, i2, i3]);
        }
    }

    mesh
}

fn guard_true_ui(
    guard: Option<&str>,
    world: &zork_core::World,
    engine: &zork_core::EngineState,
) -> bool {
    let g = match guard {
        None => return true,
        Some(s) => s.trim(),
    };
    if g.is_empty() {
        return true;
    }
    let up = g.to_uppercase();
    if up == "TRUE" {
        return true;
    }
    if up == "FALSE" || up == "FALSE-FLAG" {
        return false;
    }
    if up == "UP-CHIMNEY-FUNCTION" {
        // Mirror engine guard: allow UI button to show as available; movement enforces inventory constraints.
        return true;
    }
    if let Some(val) = engine.global_state.flags.get(&up) {
        if let Some(b) = val.as_bool() {
            return b;
        }
        if let Some(s) = val.as_str() {
            if s.eq_ignore_ascii_case("true") {
                return true;
            }
            if s.eq_ignore_ascii_case("false") {
                return false;
            }
        }
    }
    let tokens: Vec<&str> = up.split_whitespace().collect();
    if tokens.len() >= 2 {
        let obj_id = tokens[0];
        let state = tokens[1];
        if let Some(obj) = world.objects.get(obj_id) {
            let has_open = obj
                .flags
                .iter()
                .any(|f| f.eq_ignore_ascii_case("OPEN") || f.eq_ignore_ascii_case("OPENBIT"));
            match state {
                "OPEN" | "OPENBIT" => return has_open,
                "CLOSED" => return !has_open,
                _ => {}
            }
        }
    }
    if tokens.len() >= 3 && tokens[1] == "IS" {
        let obj_id = tokens[0];
        let state = tokens[2];
        if let Some(obj) = world.objects.get(obj_id) {
            let has_open = obj
                .flags
                .iter()
                .any(|f| f.eq_ignore_ascii_case("OPEN") || f.eq_ignore_ascii_case("OPENBIT"));
            match state {
                "OPEN" | "OPENBIT" => return has_open,
                "CLOSED" => return !has_open,
                _ => {}
            }
        }
    }
    false
}

fn title_case(s: &str) -> String {
    let mut out = String::new();
    for (i, part) in s
        .replace('_', " ")
        .replace('-', " ")
        .split_whitespace()
        .enumerate()
    {
        if i > 0 {
            out.push(' ');
        }
        if let Some(first) = part.chars().next() {
            out.extend(first.to_uppercase());
            out.extend(part.chars().skip(1).flat_map(|c| c.to_lowercase()));
        }
    }
    if out.is_empty() {
        s.to_string()
    } else {
        out
    }
}

fn candidate_font_dirs() -> Vec<PathBuf> {
    let candidates = [
        PathBuf::from("../Fonts"),
        PathBuf::from("../../Fonts"),
        PathBuf::from("Fonts"),
        PathBuf::from("../fonts"),
        PathBuf::from("../../fonts"),
        PathBuf::from("assets/fonts"),
        PathBuf::from("../assets/fonts"),
        PathBuf::from("../../assets/fonts"),
    ];
    candidates.into_iter().filter(|p| p.is_dir()).collect()
}

fn collect_fonts(dir: &Path, out: &mut Vec<FontEntry>) {
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                collect_fonts(&path, out);
                continue;
            }
            if let Some(ext) = path.extension() {
                if ext.eq_ignore_ascii_case("ttf") || ext.eq_ignore_ascii_case("otf") {
                    let name = path
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .unwrap_or("font")
                        .to_string();
                    out.push(FontEntry { name, path });
                }
            }
        }
    }
}

fn discover_fonts(dirs: &[PathBuf]) -> Vec<FontEntry> {
    let mut out = Vec::new();
    for dir in dirs {
        collect_fonts(dir, &mut out);
    }
    out
}

fn choose_font(fonts: &[FontEntry], hints: &[&str]) -> Option<String> {
    for hint in hints {
        if let Some(f) = fonts
            .iter()
            .find(|f| f.name.to_lowercase().contains(&hint.to_lowercase()))
        {
            return Some(f.path.to_string_lossy().to_string());
        }
    }
    fonts.first().map(|f| f.path.to_string_lossy().to_string())
}

fn load_font_data(path: &str) -> Option<FontData> {
    fs::read(path).ok().map(FontData::from_owned)
}

fn parse_flag_list(raw: &str) -> Vec<String> {
    raw.split(|c: char| c == ',' || c.is_whitespace())
        .filter(|s| !s.is_empty())
        .filter_map(|s| {
            let up = s.to_uppercase();
            if up == "NONE" {
                None
            } else {
                Some(up)
            }
        })
        .collect()
}

fn parse_flag_changes(raw: &str) -> (Vec<String>, Vec<String>) {
    let mut set_flags = Vec::new();
    let mut clear_flags = Vec::new();
    for token in raw.split(|c: char| c == ',' || c.is_whitespace()) {
        let t = token.trim();
        if t.is_empty() {
            continue;
        }
        if t.starts_with('!') || t.starts_with('-') {
            clear_flags.push(
                t.trim_start_matches(|c| c == '!' || c == '-')
                    .to_uppercase(),
            );
        } else {
            set_flags.push(t.to_uppercase());
        }
    }
    (set_flags, clear_flags)
}

fn run_command_stream_channel(
    cmd: &mut Command,
    label: &str,
    tx: Sender<String>,
) -> Result<(), String> {
    cmd.stdout(Stdio::piped());
    cmd.stderr(Stdio::piped());
    let mut child = cmd
        .spawn()
        .map_err(|e| format!("{} failed to start: {}", label, e))?;
    let mut handles = Vec::new();
    if let Some(stdout) = child.stdout.take() {
        let tx_out = tx.clone();
        let label_out = label.to_string();
        handles.push(std::thread::spawn(move || {
            use std::io::{BufRead, BufReader};
            let reader = BufReader::new(stdout);
            for line in reader.lines().flatten() {
                let _ = tx_out.send(format!("({}) {}", label_out, line));
            }
        }));
    }
    if let Some(stderr) = child.stderr.take() {
        let tx_err = tx.clone();
        let label_err = label.to_string();
        handles.push(std::thread::spawn(move || {
            use std::io::{BufRead, BufReader};
            let reader = BufReader::new(stderr);
            for line in reader.lines().flatten() {
                let _ = tx_err.send(format!("({} err) {}", label_err, line));
            }
        }));
    }
    let status = child
        .wait()
        .map_err(|e| format!("{} failed to wait: {}", label, e))?;
    for h in handles {
        let _ = h.join();
    }
    if !status.success() {
        return Err(format!("{} failed (code {:?})", label, status.code()));
    }
    Ok(())
}

fn poll_rebuild_done_marker(msg: &str) -> Option<Result<(), String>> {
    if msg == "__DONE_OK__" {
        Some(Ok(()))
    } else if let Some(rest) = msg.strip_prefix("__DONE_ERR__") {
        Some(Err(rest.trim().to_string()))
    } else {
        None
    }
}

impl ZorkEguiApp {
    fn poll_rebuild_queue(&mut self) {
        let mut msgs: Vec<String> = Vec::new();
        let mut drop_rx = false;
        if let Some(rx) = self.rebuild_rx.as_ref() {
            loop {
                match rx.try_recv() {
                    Ok(msg) => msgs.push(msg),
                    Err(TryRecvError::Empty) => break,
                    Err(TryRecvError::Disconnected) => {
                        drop_rx = true;
                        break;
                    }
                }
            }
        }
        for msg in msgs {
            if let Some(done) = poll_rebuild_done_marker(&msg) {
                drop_rx = true;
                match done {
                    Ok(()) => {
                        self.push_plain("(builder) rebuild complete");
                        if let Some(path) = self.builder_world_path.clone() {
                            match load_parser_data(&ParserFiles::with_defaults()) {
                                Ok(mut parser) => {
                                    if let Ok(world) = load_world(&path) {
                                        parser.world = Some(world);
                                        self.parser = Some(parser);
                                        self.refresh_world_views();
                                        self.push_plain("(builder) reloaded parser assets");
                                    } else {
                                        self.push_plain(format!(
                                            "(builder) reload world failed: {}",
                                            path.display()
                                        ));
                                    }
                                }
                                Err(e) => self
                                    .push_plain(format!("(builder) reload parser failed: {}", e)),
                            }
                        }
                    }
                    Err(e) => self.push_plain(format!("(builder) rebuild failed: {}", e)),
                }
            } else {
                self.push_plain(msg);
            }
        }
        if drop_rx {
            self.rebuild_rx = None;
        }
    }
}

#[cfg(feature = "speech_input")]
fn spawn_speech_runtime(
    model_path: &str,
    language: Option<String>,
) -> Result<(Sender<SpeechCommand>, Receiver<SpeechEvent>), String> {
    if !Path::new(model_path).exists() {
        return Err(format!("Model not found at {}", model_path));
    }
    let (cmd_tx, cmd_rx) = mpsc::channel::<SpeechCommand>();
    let (evt_tx, evt_rx) = mpsc::channel::<SpeechEvent>();
    thread::spawn(move || {
        let mut stream: Option<Stream> = None;
        let mut last_sr: u32 = 16_000;
        let mut last_channels: u16;
        let mut ctx: Option<WhisperContext> = None;
        let mut ctx_model: Option<String> = None;
        let mut lang_opt = language.clone();
        let audio_buf = Arc::new(Mutex::new(Vec::<f32>::new()));
        const MAX_SAMPLES: usize = 16_000 * 120; // cap ~2 minutes
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
                Err(e) => Err(format!("Failed to load model: {e}")),
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

        let host = cpal::default_host();
        while let Ok(cmd) = cmd_rx.recv() {
            match cmd {
                SpeechCommand::Start {
                    model_path: path,
                    language,
                } => {
                    lang_opt = language;
                    if let Some(mut buf) = audio_buf.lock().ok() {
                        buf.clear();
                    }
                    if stream.is_some() {
                        let _ = evt_tx.send(SpeechEvent::RecordingStarted);
                        continue;
                    }
                    let device = match host.default_input_device() {
                        Some(d) => d,
                        None => {
                            let _ =
                                evt_tx.send(SpeechEvent::Error("No input device available".into()));
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
                                        let mut guard = match buf_clone.lock() {
                                            Ok(g) => g,
                                            Err(_) => return,
                                        };
                                        for frame in data.chunks(last_channels as usize) {
                                            let mut sum = 0.0f32;
                                            for s in frame {
                                                sum += *s;
                                            }
                                            guard.push(sum / last_channels as f32);
                                            if guard.len() > MAX_SAMPLES {
                                                let drop = guard.len() - MAX_SAMPLES;
                                                guard.drain(0..drop);
                                            }
                                        }
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
                                        let mut guard = match buf_clone.lock() {
                                            Ok(g) => g,
                                            Err(_) => return,
                                        };
                                        for frame in data.chunks(last_channels as usize) {
                                            let mut sum = 0.0f32;
                                            for s in frame {
                                                sum += *s as f32 / i16::MAX as f32;
                                            }
                                            guard.push(sum / last_channels as f32);
                                            if guard.len() > MAX_SAMPLES {
                                                let drop = guard.len() - MAX_SAMPLES;
                                                guard.drain(0..drop);
                                            }
                                        }
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
                                        let mut guard = match buf_clone.lock() {
                                            Ok(g) => g,
                                            Err(_) => return,
                                        };
                                        for frame in data.chunks(last_channels as usize) {
                                            let mut sum = 0.0f32;
                                            for s in frame {
                                                sum += (*s as f32 / u16::MAX as f32) * 2.0 - 1.0;
                                            }
                                            guard.push(sum / last_channels as f32);
                                            if guard.len() > MAX_SAMPLES {
                                                let drop = guard.len() - MAX_SAMPLES;
                                                guard.drain(0..drop);
                                            }
                                        }
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
                    if let Err(e) = load_ctx(&path, &mut ctx_model, &mut ctx) {
                        let _ = evt_tx.send(SpeechEvent::Error(e));
                    }
                }
                SpeechCommand::Stop | SpeechCommand::Cancel => {
                    if stream.is_some() {
                        stream = None;
                        let _ = evt_tx.send(SpeechEvent::RecordingStopped);
                    }
                    let samples = {
                        if let Ok(mut guard) = audio_buf.lock() {
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

#[cfg(not(feature = "speech_input"))]
fn spawn_speech_runtime(
    _model_path: &str,
    _language: Option<String>,
) -> Result<(Sender<SpeechCommand>, Receiver<SpeechEvent>), String> {
    Err("Speech input not enabled in this build".into())
}

#[cfg(feature = "speech_input")]
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

fn format_lint_issue(issue: &LintIssue) -> String {
    let tag = match issue.level {
        LintLevel::Error => "ERROR",
        LintLevel::Warn => "WARN",
        LintLevel::Info => "INFO",
    };
    format!("[{}] {}", tag, issue.message)
}

fn normalize_room_id(raw: &str) -> String {
    let trimmed = raw.trim().trim_matches('"');
    let replaced: String = trimmed
        .chars()
        .map(|c| {
            if c.is_whitespace() || c == '_' {
                '-'
            } else {
                c
            }
        })
        .collect();
    replaced.to_uppercase()
}
