# Changelog

All notable changes to Idol Studio are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses [Semantic Versioning](https://semver.org/).

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
