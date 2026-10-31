## Development Workflow

**IMPORTANT**: After any code change, bug fix, or feature addition/removal, you MUST complete all of these steps:

1. **Update README.md** if the change affects:
   - Usage examples or commands
   - Installation instructions
   - Configuration options
   - Available features

2. **Update CHANGELOG.md**:
   - Add new version number following semantic versioning (MAJOR.MINOR.PATCH)
   - Add entry under appropriate category (Added, Changed, Fixed, Removed)
   - Include date in format YYYY-MM-DD

3. **Bump the version** to match the new CHANGELOG.md entry, in all of these:
   - `src-tauri/Cargo.toml` (`version = "..."`). This is the version the app and the release use.
   - `package.json` (`"version"`)
   - Then run `cargo check --manifest-path src-tauri/Cargo.toml` so `Cargo.lock` picks up the new version.

4. **Commit and push changes**:
   - Use `git add` to stage all modified files (README.md, CHANGELOG.md, and code files)
   - Create descriptive commit message following existing style
   - Push to remote repository with `git push`

5. **Publish the .exe**: pushing a version bump to `master` makes the Release workflow (`.github/workflows/release.yml`) build the app and publish a GitHub release `v<version>`. The release has `IdolStudio_<version>_x64-setup.exe` (installs the CLI and adds its folder to the user PATH), `IdolStudio_<version>_x64-portable.exe` and `IdolStudio_<version>_x64-cli.exe` attached, and its notes come from the CHANGELOG.md section.
   - Check that it succeeded: `gh run list --workflow release.yml --limit 1`, then `gh release view v<version>`.
   - The app's update checker reads these releases, so every version bump must end up as a release.

6. **Run `cargo clean` after pushing**: `cargo clean --manifest-path src-tauri/Cargo.toml`. Tauri build output is several GB.

## Project notes

- To generate splats yourself (testing, demos), use the CLI rather than driving the GUI: `src-tauri/target/<profile>/idol-studio-cli.exe generate <image> --json`. Run `serve` first when generating more than once. `idol-studio-cli --help` lists everything.
- Node.js comes from fnm. Load it in the same command as npm, e.g. in PowerShell: `fnm env --use-on-cd --shell powershell | Out-String | Invoke-Expression; npm run build`.
- `src-tauri/python/idol_worker.py` and `idol_export.py` are embedded in the binary with `include_str!`. Changing either requires rebuilding the Rust side.
- Test exports through the CLI, not the GUI: `idol-studio-cli export result.ply --format glb,spz,stl --out <dir> --json`. Check `.spz` with an SPZ viewer and `.stl` in a slicer; mesh colours and normals matter.
- The worker must never leave a blocking read pending on stdin on Windows: DLL loads (`import torch`) hang while one is pending. Keep the `PeekNamedPipe` reader.
- Keep stdout of the worker for JSON events only. Everything else goes to stderr.
- Never commit `.env.local` (it holds the Hugging Face token).
