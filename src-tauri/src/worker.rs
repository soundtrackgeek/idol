//! Runs the Python worker (python/idol_worker.py) as a long-lived child process.
//!
//! Requests go to its stdin as JSON lines. Its stdout (JSON events) is forwarded to the UI as `worker-event`,
//! its stderr (library logs) as `worker-log`, and its exit as `worker-exit`.

use crate::settings;
use serde_json::Value;
use std::io::{BufRead, BufReader, Read, Write};
use std::path::PathBuf;
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::Mutex;
use tauri::{AppHandle, Emitter, Manager, State};

#[derive(Default)]
pub struct Worker {
    child: Option<Child>,
    stdin: Option<ChildStdin>,
}

#[derive(Default)]
pub struct WorkerState(pub Mutex<Worker>);

/// Hide the console window that Windows would open for a console program.
pub fn no_window(cmd: &mut Command) {
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }
    #[cfg(not(windows))]
    let _ = cmd;
}

/// The worker script is compiled into the app and written to the app data folder, so the installer and the
/// portable .exe behave the same.
const WORKER_SOURCE: &str = include_str!("../python/idol_worker.py");

pub fn script_path(app: &AppHandle) -> Result<PathBuf, String> {
    let dir = app.path().app_local_data_dir().map_err(|e| e.to_string())?.join("worker");
    let path = dir.join("idol_worker.py");
    let current = std::fs::read_to_string(&path).unwrap_or_default();
    if current != WORKER_SOURCE {
        std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
        std::fs::write(&path, WORKER_SOURCE).map_err(|e| format!("could not write {}: {e}", path.display()))?;
    }
    Ok(path)
}

impl Worker {
    fn running(&mut self) -> bool {
        match self.child.as_mut() {
            Some(child) => matches!(child.try_wait(), Ok(None)),
            None => false,
        }
    }

    fn start(&mut self, app: &AppHandle) -> Result<(), String> {
        if self.running() {
            return Ok(());
        }
        let settings = settings::load(app);
        let python = settings::python(app, &settings)?;
        if !python.exists() {
            return Err(format!(
                "Python was not found at {}. Open Settings and run \"Set up environment\", or choose a Python that has Idol installed.",
                python.display()
            ));
        }
        let script = script_path(app)?;
        let mut cmd = Command::new(&python);
        cmd.arg("-u")
            .arg(&script)
            .env("PYTHONUNBUFFERED", "1")
            .env("PYTHONIOENCODING", "utf-8")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        if !settings.hf_token.trim().is_empty() {
            cmd.env("HF_TOKEN", settings.hf_token.trim());
        }
        no_window(&mut cmd);
        let mut child = cmd
            .spawn()
            .map_err(|e| format!("could not start {}: {e}", python.display()))?;

        let stdout = child.stdout.take().ok_or("no stdout")?;
        let stderr = child.stderr.take().ok_or("no stderr")?;
        self.stdin = child.stdin.take();
        let pid = child.id();
        self.child = Some(child);

        let events = app.clone();
        std::thread::spawn(move || {
            for line in BufReader::new(stdout).lines() {
                let Ok(line) = line else { break };
                match serde_json::from_str::<Value>(&line) {
                    Ok(value) => {
                        let _ = events.emit("worker-event", value);
                    }
                    Err(_) => {
                        let _ = events.emit("worker-log", line);
                    }
                }
            }
            // stdout closed: the process ended.
            let code = events
                .state::<WorkerState>()
                .0
                .lock()
                .ok()
                .and_then(|mut worker| {
                    // Only clean up if this is still the same process (it may have been replaced).
                    if worker.child.as_ref().map(|c| c.id()) != Some(pid) {
                        return None;
                    }
                    let code = worker.child.as_mut().and_then(|c| c.wait().ok()).and_then(|s| s.code());
                    worker.child = None;
                    worker.stdin = None;
                    code
                });
            let _ = events.emit("worker-exit", code);
        });

        let logs = app.clone();
        std::thread::spawn(move || forward_log(stderr, &logs));
        let _ = app.emit("worker-started", python.display().to_string());
        Ok(())
    }

    fn send(&mut self, app: &AppHandle, request: &Value) -> Result<(), String> {
        self.start(app)?;
        let stdin = self.stdin.as_mut().ok_or("the worker has no input")?;
        let mut line = serde_json::to_string(request).map_err(|e| e.to_string())?;
        line.push('\n');
        stdin
            .write_all(line.as_bytes())
            .and_then(|_| stdin.flush())
            .map_err(|e| format!("could not talk to the worker: {e}"))
    }

    pub fn stop(&mut self) {
        if let Some(stdin) = self.stdin.as_mut() {
            let _ = stdin.write_all(b"{\"cmd\":\"quit\"}\n");
            let _ = stdin.flush();
        }
        self.stdin = None;
        if let Some(mut child) = self.child.take() {
            // Give it a moment to exit cleanly, then make sure it is gone (frees the GPU memory).
            for _ in 0..20 {
                if matches!(child.try_wait(), Ok(Some(_))) {
                    return;
                }
                std::thread::sleep(std::time::Duration::from_millis(100));
            }
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

/// Forward stderr as log lines. tqdm and download bars use '\r' to redraw, so split on that too.
fn forward_log(stream: impl Read, app: &AppHandle) {
    let mut reader = BufReader::new(stream);
    let mut buffer = Vec::new();
    let mut byte = [0u8; 1];
    let mut last_was_cr = false;
    loop {
        match reader.read(&mut byte) {
            Ok(0) | Err(_) => break,
            Ok(_) => {
                if byte[0] == b'\n' || byte[0] == b'\r' {
                    if !buffer.is_empty() {
                        let text = String::from_utf8_lossy(&buffer).trim_end().to_string();
                        let _ = app.emit(if last_was_cr { "worker-log-replace" } else { "worker-log" }, text);
                        buffer.clear();
                    }
                    last_was_cr = byte[0] == b'\r';
                } else {
                    buffer.push(byte[0]);
                }
            }
        }
    }
    if !buffer.is_empty() {
        let _ = app.emit("worker-log", String::from_utf8_lossy(&buffer).to_string());
    }
}

#[tauri::command]
pub fn worker_send(app: AppHandle, state: State<'_, WorkerState>, request: Value) -> Result<(), String> {
    state.0.lock().map_err(|e| e.to_string())?.send(&app, &request)
}

#[tauri::command]
pub fn worker_stop(state: State<'_, WorkerState>) -> Result<(), String> {
    state.0.lock().map_err(|e| e.to_string())?.stop();
    Ok(())
}

#[tauri::command]
pub fn worker_running(state: State<'_, WorkerState>) -> Result<bool, String> {
    Ok(state.0.lock().map_err(|e| e.to_string())?.running())
}
