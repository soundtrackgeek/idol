# Changelog

All notable changes to Idol Studio are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses [Semantic Versioning](https://semver.org/).

## [0.5.0] - 2026-10-31

### Added

- **Game-ready .glb export** for game engines and rigging: a low-poly mesh (20k triangles by default, 5k to 50k in the app) with automatic UVs (xatlas) and the splat colours baked into a texture (2048 px by default, 1024 or 4096 in the app), smooth normals and a matte PBR material. In the Export dialog under **Game-ready**; on the command line `--format game-glb` with `--game-faces` and `--texture-size`. Game-ready meshes preview in the viewer with their texture.
- Environment setup installs `xatlas` for UV unwrapping.

### Fixed

- Mesh exports are closed where the object is solid. Removing the skin Poisson stretches across gaps (between legs) left the inner side of those parts open, which showed as holes in single-sided viewers and game engines.
- Thin separate pieces of a mesh (for example the top of a lamp's base) could face inward, so single-sided viewers showed a hole there. Each piece is now oriented using the splats: outside is the side with empty space.
- Mesh colours are sampled just outside the surface, so the darker interior splats of solid parts no longer darken them.

## [0.4.1] - 2026-10-31

### Fixed

- Mesh exports had holes on the sides Idol invents (for example the back of a figure's head) and around feet and hands. Those sides are made of fainter splats, which the Poisson mesh dropped, and a density-based trim then cut the thin areas open. The mesh now keeps faint splats and only removes triangles that are far from any splat, so it stays closed where the object is and still separates parts with real gaps between them (legs).
- Mesh exports no longer contain dozens to hundreds of loose fragments; small pieces left after simplification are removed.
- The mesh preview in the viewer draws both sides of each triangle, so small cracks no longer show the background through the model.

## [0.4.0] - 2026-10-31

### Added

- **Much better mesh export.** Meshes now default to screened Poisson reconstruction (Open3D) from the Gaussian centres and their smallest-axis normals, instead of the opacity density field. Faces, eyes, hands and thin parts survive, and vertex colours are sampled from the nearest splats instead of a blurred voxel average.
- A **Reconstruction** choice in the Export dialog: Poisson (recommended) or density field (for very thin/unusual objects). The CLI has `--method poisson|density`; `--resolution` controls Poisson detail.
- GLB exports now include a matte PBR material. Without one, glTF viewers use the spec's metallic default, which was washing out the vertex colours.
- GLB vertex colours are converted from the splat's display (sRGB) values to glTF's linear space. Blender, Windows 3D Viewer and other compliant viewers no longer show a pale, washed-out model.

### Changed

- Environment setup installs Open3D for Poisson reconstruction.
- Mesh smoothing default is 3 passes instead of 10; high smoothing was melting details. The density method still uses its surface-threshold control.

## [0.3.0] - 2026-10-31

### Added

- **Export results** from the app's Export button or the CLI, to:
  - splats: `.spz` (compressed), `.splat` and a standard `.ply` copy;
  - a point cloud: `.ply` with colours;
  - meshes with vertex colours: `.glb`, `.obj`, `.ply` and `.stl` for 3D printing.
- Mesh options: grid resolution, surface threshold, smoothing, target triangle count and an optional real-world height (e.g. 100 mm for an STL). Exported `.glb` meshes can be previewed in the viewer with lighting.
- Exported files are remembered on each result (chips on the result card) and can be picked from the viewer's Splat/Mesh menu.
- `idol-studio-cli export` and `generate --export`, with the same mesh options. `serve` handles exports too.
- The installer now adds the CLI's folder to the user PATH (and removes it on uninstall); Settings → Command line shows whether it is on PATH, with Add/Remove buttons. A portable build with the CLI next to it uses the same toggle.
- `idol-studio-cli path add|remove|status`.

### Changed

- Environment setup also installs `scipy`, `scikit-image`, `trimesh` and `fast-simplification` for mesh export. Existing environments get them from **Set up / repair environment** (or `idol-studio-cli setup`).
- Exported meshes are upright (y-up for glTF/OBJ/PLY, z-up for STL), centred and standing on the ground plane.

### Fixed

- Fast orbiting no longer makes the splat look hollow or smeared. Spark sorts splats asynchronously; the viewer now draws only after the sort matches the exact camera pose, instead of showing frames with an order a few frames old (measured 0° sort lag during a fast spin, down from 5–6°).
- Mesh exports have outward-facing normals and sample colour from the visible surface, not the object's interior, so they light and colour correctly.
- `.spz` exports are converted from the PLY frame to SPZ's y-up frame, so they load upright in SPZ viewers.
- The frontend build now has the Vite client types it needs for `import.meta.env`.

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
