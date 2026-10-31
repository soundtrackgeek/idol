//! One-click Python environment: a venv with CUDA PyTorch, Idol and Triton (for torch.compile).
//!
//! Uses `uv` when it is installed (fast, and it can fetch Python 3.12 by itself), otherwise a Python found
//! on the system plus pip. Progress is streamed to the UI as `setup-log` lines and a final `setup-done`.

use crate::settings;
use crate::worker::no_window;
use serde::Serialize;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use tauri::{AppHandle, Emitter};

const TORCH_INDEX: &str = "https://download.pytorch.org/whl/cu128";
const IDOL_PACKAGE: &str = "idol @ https://github.com/speridlabs/idol-sdk/archive/refs/heads/main.zip";

static RUNNING: AtomicBool = AtomicBool::new(false);

#[derive(Serialize, Clone)]
struct SetupDone {
    ok: bool,
    message: String,
    python: String,
}

fn log(app: &AppHandle, line: impl Into<String>) {
    let _ = app.emit("setup-log", line.into());
}

fn found(program: &str, args: &[&str]) -> bool {
    let mut cmd = Command::new(program);
    cmd.args(args).stdout(Stdio::null()).stderr(Stdio::null());
    no_window(&mut cmd);
    cmd.status().map(|s| s.success()).unwrap_or(false)
}

/// Run a command, streaming its output lines to the setup log.
fn run(app: &AppHandle, program: &Path, args: &[&str]) -> Result<(), String> {
    log(app, format!("> {} {}", program.display(), args.join(" ")));
    let mut cmd = Command::new(program);
    cmd.args(args)
        .env("PYTHONIOENCODING", "utf-8")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    no_window(&mut cmd);
    let mut child = cmd.spawn().map_err(|e| format!("could not run {}: {e}", program.display()))?;
    let stderr = child.stderr.take();
    let app_err = app.clone();
    let err_thread = std::thread::spawn(move || {
        if let Some(stderr) = stderr {
            for line in BufReader::new(stderr).lines().map_while(Result::ok) {
                log(&app_err, line);
            }
        }
    });
    if let Some(stdout) = child.stdout.take() {
        for line in BufReader::new(stdout).lines().map_while(Result::ok) {
            log(app, line);
        }
    }
    let _ = err_thread.join();
    let status = child.wait().map_err(|e| e.to_string())?;
    if status.success() {
        Ok(())
    } else {
        Err(format!("{} failed ({status})", program.display()))
    }
}

fn setup(app: &AppHandle) -> Result<PathBuf, String> {
    let env = settings::env_dir(app)?;
    let env_str = env.to_string_lossy().to_string();
    let python = settings::env_python(&env);
    let python_str = python.to_string_lossy().to_string();
    if let Some(parent) = env.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    log(app, format!("Environment folder: {env_str}"));

    if found("uv", &["--version"]) {
        log(app, "Using uv.");
        let uv = PathBuf::from("uv");
        run(app, &uv, &["venv", "--allow-existing", "--python", "3.12", &env_str])?;
        log(app, "Installing PyTorch with CUDA 12.8 (about 3 GB, this takes a while)...");
        run(app, &uv, &["pip", "install", "--python", &python_str, "torch", "torchvision", "--index-url", TORCH_INDEX])?;
        log(app, "Installing Idol, Triton and accelerate...");
        let mut args = vec!["pip", "install", "--python", &python_str, IDOL_PACKAGE, "accelerate"];
        if cfg!(windows) {
            args.push("triton-windows");
        }
        run(app, &uv, &args)?;
    } else {
        // No uv: find a Python 3.10-3.12 to create the venv with.
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
        log(app, "uv not found, using Python + pip.");
        let mut args = prefix.clone();
        args.extend(["-m", "venv", &env_str]);
        run(app, &base, &args)?;
        run(app, &python, &["-m", "pip", "install", "--upgrade", "pip"])?;
        log(app, "Installing PyTorch with CUDA 12.8 (about 3 GB, this takes a while)...");
        run(app, &python, &["-m", "pip", "install", "torch", "torchvision", "--index-url", TORCH_INDEX])?;
        log(app, "Installing Idol, Triton and accelerate...");
        let mut args = vec!["-m", "pip", "install", IDOL_PACKAGE, "accelerate"];
        if cfg!(windows) {
            args.push("triton-windows");
        }
        run(app, &python, &args)?;
    }

    run(
        app,
        &python,
        &["-c", "import torch, idol; print('PyTorch', torch.__version__, '| CUDA', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no GPU')"],
    )?;
    Ok(python)
}

/// Create or update the managed environment. Returns immediately; listen for `setup-log` / `setup-done`.
#[tauri::command]
pub fn setup_environment(app: AppHandle) -> Result<(), String> {
    if RUNNING.swap(true, Ordering::SeqCst) {
        return Err("setup is already running".into());
    }
    std::thread::spawn(move || {
        let result = setup(&app);
        RUNNING.store(false, Ordering::SeqCst);
        let done = match result {
            Ok(python) => SetupDone {
                ok: true,
                message: "Environment is ready.".into(),
                python: python.display().to_string(),
            },
            Err(message) => SetupDone { ok: false, message, python: String::new() },
        };
        let _ = app.emit("setup-done", done);
    });
    Ok(())
}
