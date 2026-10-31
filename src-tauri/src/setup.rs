//! One-click Python environment: a venv with CUDA PyTorch, Idol, Triton (for torch.compile) and accelerate.
//!
//! Uses `uv` when it is installed (fast, and it can fetch Python 3.12 by itself), otherwise a Python found
//! on the system plus pip. Used by the GUI (streamed as `setup-log` / `setup-done`) and `idol-studio-cli setup`.

use crate::process::no_window;
use crate::settings;
use serde::Serialize;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use tauri::{AppHandle, Emitter};

const TORCH_INDEX: &str = "https://download.pytorch.org/whl/cu128";
const IDOL_PACKAGE: &str = "idol @ https://github.com/speridlabs/idol-sdk/archive/refs/heads/main.zip";

static RUNNING: AtomicBool = AtomicBool::new(false);

pub type Log<'a> = &'a (dyn Fn(String) + Sync);

fn found(program: &str, args: &[&str]) -> bool {
    let mut cmd = Command::new(program);
    cmd.args(args).stdout(Stdio::null()).stderr(Stdio::null());
    no_window(&mut cmd);
    cmd.status().map(|s| s.success()).unwrap_or(false)
}

/// Run a command, streaming its output lines to the log.
fn run(log: Log, program: &Path, args: &[&str]) -> Result<(), String> {
    log(format!("> {} {}", program.display(), args.join(" ")));
    let mut cmd = Command::new(program);
    cmd.args(args)
        .env("PYTHONIOENCODING", "utf-8")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    no_window(&mut cmd);
    let mut child = cmd.spawn().map_err(|e| format!("could not run {}: {e}", program.display()))?;
    let stderr = child.stderr.take();
    let stdout = child.stdout.take();
    std::thread::scope(|scope| {
        scope.spawn(|| {
            if let Some(stderr) = stderr {
                for line in BufReader::new(stderr).lines().map_while(Result::ok) {
                    log(line);
                }
            }
        });
        if let Some(stdout) = stdout {
            for line in BufReader::new(stdout).lines().map_while(Result::ok) {
                log(line);
            }
        }
    });
    let status = child.wait().map_err(|e| e.to_string())?;
    if status.success() {
        Ok(())
    } else {
        Err(format!("{} failed ({status})", program.display()))
    }
}

/// Create or update the managed environment. Returns the environment's python.
pub fn setup(log: Log) -> Result<PathBuf, String> {
    if RUNNING.swap(true, Ordering::SeqCst) {
        return Err("setup is already running".into());
    }
    let result = setup_inner(log);
    RUNNING.store(false, Ordering::SeqCst);
    result
}

fn setup_inner(log: Log) -> Result<PathBuf, String> {
    let env = settings::env_dir()?;
    let env_str = env.to_string_lossy().to_string();
    let python = settings::env_python(&env);
    let python_str = python.to_string_lossy().to_string();
    if let Some(parent) = env.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    log(format!("Environment folder: {env_str}"));
    // accelerate: faster model loading; the rest: mesh export (marching cubes, mesh I/O, decimation).
    let mut extras = vec![IDOL_PACKAGE, "accelerate", "scipy", "scikit-image", "trimesh", "fast-simplification"];
    if cfg!(windows) {
        extras.push("triton-windows");
    }

    if found("uv", &["--version"]) {
        log("Using uv.".into());
        let uv = PathBuf::from("uv");
        run(log, &uv, &["venv", "--allow-existing", "--python", "3.12", &env_str])?;
        log("Installing PyTorch with CUDA 12.8 (about 3 GB, this takes a while)...".into());
        run(log, &uv, &["pip", "install", "--python", &python_str, "torch", "torchvision", "--index-url", TORCH_INDEX])?;
        log("Installing Idol, Triton and the export packages...".into());
        let mut args = vec!["pip", "install", "--python", &python_str];
        args.extend(&extras);
        run(log, &uv, &args)?;
    } else {
        // No uv: find a Python 3.10-3.13 to create the venv with.
        let base = if cfg!(windows) && found("py", &["-3.12", "--version"]) {
            Some((PathBuf::from("py"), vec!["-3.12"]))
        } else if found("python", &["-c", "import sys; assert (3, 10) <= sys.version_info[:2] <= (3, 13)"]) {
            Some((PathBuf::from("python"), vec![]))
        } else {
            None
        };
        let Some((base, prefix)) = base else {
            return Err("Neither uv nor Python 3.10-3.13 was found. Install uv (https://docs.astral.sh/uv/) \
                        and try again; it downloads the right Python by itself."
                .into());
        };
        log("uv not found, using Python + pip.".into());
        let mut args = prefix.clone();
        args.extend(["-m", "venv", &env_str]);
        run(log, &base, &args)?;
        run(log, &python, &["-m", "pip", "install", "--upgrade", "pip"])?;
        log("Installing PyTorch with CUDA 12.8 (about 3 GB, this takes a while)...".into());
        run(log, &python, &["-m", "pip", "install", "torch", "torchvision", "--index-url", TORCH_INDEX])?;
        log("Installing Idol, Triton and the export packages...".into());
        let mut args = vec!["-m", "pip", "install"];
        args.extend(&extras);
        run(log, &python, &args)?;
    }

    run(
        log,
        &python,
        &["-c", "import torch, idol; print('PyTorch', torch.__version__, '| CUDA', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no GPU')"],
    )?;
    Ok(python)
}

#[derive(Serialize, Clone)]
struct SetupDone {
    ok: bool,
    message: String,
    python: String,
}

/// GUI command: returns immediately; listen for `setup-log` / `setup-done`.
#[tauri::command]
pub fn setup_environment(app: AppHandle) -> Result<(), String> {
    if RUNNING.load(Ordering::SeqCst) {
        return Err("setup is already running".into());
    }
    std::thread::spawn(move || {
        let emit = |line: String| {
            let _ = app.emit("setup-log", line);
        };
        let done = match setup(&emit) {
            Ok(python) => SetupDone { ok: true, message: "Environment is ready.".into(), python: python.display().to_string() },
            Err(message) => SetupDone { ok: false, message, python: String::new() },
        };
        let _ = app.emit("setup-done", done);
    });
    Ok(())
}
