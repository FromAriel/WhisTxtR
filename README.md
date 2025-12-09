# WhisTxtR (Whisper Tray) – Build & Run

Resident Windows tray app that records on hold-to-talk (`Ctrl+Shift`), runs Whisper locally (CPU or CUDA), and injects the transcript into the active app. Overlay shows recording/processing and live partial text.

## Prerequisites (Windows 10/11)
- Rust (stable, MSVC toolchain).
- Visual Studio 2022 Build Tools (MSVC, Windows 10 SDK).
- CMake (tested path: `G:\Unity Projects\Cmake\bin\cmake.exe`).
- CUDA 12.4 for GPU builds (install toolkit; tested at `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4`).
- libclang for bindgen (tested at `J:\llvm\bin` with `LIBCLANG_PATH` set there).

## Quick env setup (PowerShell example)
```powershell
# Adjust paths to your machine
$env:CMAKE           = 'G:\Unity Projects\Cmake\bin\cmake.exe'
$env:CMAKE_GENERATOR = 'Visual Studio 17 2022'
$env:LIBCLANG_PATH   = 'J:\llvm\bin'
$env:CUDA_PATH             = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4'
$env:CUDA_PATH_V12_4       = $env:CUDA_PATH
$env:CUDA_TOOLKIT_ROOT_DIR = $env:CUDA_PATH
$env:CUDACXX               = "$env:CUDA_PATH\bin\nvcc.exe"
$env:CudaToolkitDir        = $env:CUDA_PATH
$env:PATH = "$env:CUDA_PATH\bin;$env:CUDA_PATH\lib\x64;$env:PATH"
```
Verify: `where nvcc`, `cmake --version`, `where clang` (or `where libclang.dll`).

## Build
- CPU: `cargo build --release`
- CUDA: `cargo build --release --features cuda`

Outputs land in `rust/target/release/stt-whis-tray.exe`.

## Run
From repo root:
```powershell
cd J:\whistxt\rust
.\target\release\stt-whis-tray.exe
```
Config lives at `%APPDATA%\WhisTray\config.json` (created on first run).

## Models (not in repo)
- Default expected path: J:/whistxt/models/[ggml-small-q5_1.bin](https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.en-q5_1.bin) (set in config).
- Models are **not** committed; place your `.bin`/`.gguf` under `models/` and update config if needed.
- Provide/download your own and verify checksum; add a small download script if distributing externally.

## Hotkey & behavior
- Hold `Ctrl+Shift` to record; release to transcribe and inject.
- Overlay: topmost bar with red blink; shows “Listening…”, live partial text during processing, then final text.
- Injection: defaults to clipboard+Ctrl+V. Config is in `%APPDATA%\WhisTray\config.json` (`inject_mode`, `append_newline`, `model_path`, `use_cuda`, etc.).
- Tray menu: right-click tray icon → Quit, toggle “Start with Windows” (startup link).

## Troubleshooting
- Missing CUDA DLLs: ensure `CUDA_PATH/bin` and `CUDA_PATH/lib/x64` are on `PATH`; correct `CUDA_TOOLKIT_ROOT_DIR`/`CUDACXX`.
- Bindgen/libclang errors: set `LIBCLANG_PATH` to your LLVM `bin` folder.
- Model not found: confirm the path in `%APPDATA%\WhisTray\config.json` exists.
- Partial overlay not updating: ensure the app is running from the same directory as the model and CUDA DLLs are visible on `PATH`; overlay draws via Win32/GDI (no GPU UI dependency). Use the log output in the console to confirm state changes.

## What’s not included
- Model binaries (large): ignored by `.gitignore`.
- Any network calls: audio and text stay on-device.

License: MIT (matches Cargo.toml).
