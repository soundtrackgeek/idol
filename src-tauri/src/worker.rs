//! Runs the Python worker (python/idol_worker.py) as a long-lived child process.
//!
//! Requests go to its stdin as JSON lines. Its stdout (JSON events) is forwarded to the UI as `worker-event`,
//! its stderr (library logs) as `worker-log`, and its exit as `worker-exit`.

use crate::{process, settings};
use serde_json::Value;
use std::io::{BufRead, BufReader, Read, Write};
use std::process::{Child, ChildStdin};
use std::sync::Mutex;
use tauri::{AppHandle, Emitter, Manager, State};

#[derive(Default)]
pub struct Worker {
    child: Option<Child>,
    stdin: Option<ChildStdin>,
}

#[derive(Default)]
pub struct WorkerState(pub Mutex<Worker>);

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
        let settings = settings::load();
        let (mut child, python) = process::spawn_worker(&settings)?;

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
