# Changelog

All notable changes to Idol Studio are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses [Semantic Versioning](https://semver.org/).

## [0.2.0] - 2026-10-31

### Added

- Command line interface for scripts and AI agents: `idol-studio-cli.exe`. It is installed next to the app and published as `IdolStudio_<version>_x64-cli.exe`. The app exe takes the same commands.
  - Commands: `generate`, `preview`, `check-access`, `info`, `setup`, `serve`, `config`, `library`.
  - `generate` takes every Idol option, files and folders, variants and random seeds.
  - `--json` gives JSON-lines output: progress, results with paths, camera and timings, and a summary. Exit codes tell errors and missing Hugging Face access apart.
- `serve` keeps the model loaded on 127.0.0.1:47817. Later CLI calls use it automatically and skip the model load and compile.
- Results generated from the command line appear in the open app right away.
- Settings → Command line shows the CLI path.

### Changed

- The image library is now stored in `library.json` in the app config folder (shared with the CLI) instead of the webview. The old library is imported automatically.
- Python workers stop when the app or CLI that started them exits, even if it is killed or crashes, so no GPU memory is left behind.
- The worker always answers a `load` request, even when the model is already loaded.

## [0.1.1] - 2026-10-31

### Fixed

- Splats loaded upside down in the viewer. Idol saves y-up objects, while Spark uses the y-down convention of captured 3DGS scenes, so the splat is now turned upright by default. "Flip" undoes that for files in the other convention.
- A result that finished after the page reloaded mid-run was lost. The worker now reports the image and settings with each result, so the result is still filed under its image.

### Changed

- The viewer opens a splat from Idol's predicted input camera, so you first see it from the same angle as the source image.
- Environment setup also installs `accelerate`, for faster model loading with less RAM.

## [0.1.0] - 2026-10-31

### Added

- Tauri 2 desktop app for Idol (single image to 3D Gaussian splat).
- Image library: add images by file picker, folder or drag and drop. Kept between sessions.
- All of Idol's options:
  - mode, crop margin
  - seed with random seeds, and variants (consecutive seeds)
  - number of Gaussians, steps, guidance and time shift
  - DINOv3 precision, faster sampling (torch.compile)
  - weights (Hugging Face repo or local folder)
- Generation queue with live sampling progress, cancel, "Generate all", unload model and stop worker.
- "Preview crop" shows the background-removed, cropped model input. It loads only BiRefNet, so it works before the full model is loaded.
- 3D splat viewer (Spark + three.js) with orbit, auto-rotate, flip, background and reset view. Tabs for the source image and the model input.
- Results per image: open the `.ply`, show in folder, reuse its settings, delete.
- Python worker that keeps the Idol pipeline loaded on the GPU and talks JSON lines over stdin/stdout.
- One-click environment setup (uv or pip): CUDA 12.8 PyTorch, Idol and triton-windows.
- Hugging Face token setting and an access check for the gated encoders.
- Update check against GitHub Releases on start, every 30 minutes and on demand, with download and install.
- GitHub Actions workflow that publishes the installer and portable exe as a release when the version is bumped.
