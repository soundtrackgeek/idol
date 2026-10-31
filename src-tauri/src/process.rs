//! Starting the Python worker, shared by the GUI and the CLI.

use crate::{paths, settings};
use settings::Settings;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};

/// The worker script is compiled into the app and written to the app data folder, so the installer, the portable
/// .exe and the CLI all behave the same.
const WORKER_SOURCE: &str = include_str!("../python/idol_worker.py");

pub fn script_path() -> Result<PathBuf, String> {
    let dir = paths::local_data_dir()?.join("worker");
    let path = dir.join("idol_worker.py");
    let current = std::fs::read_to_string(&path).unwrap_or_default();
    if current != WORKER_SOURCE {
        std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
        // Write to a temp file first: another process may be starting the same script right now.
        paths::write_atomic(&path, WORKER_SOURCE.as_bytes())?;
    }
    Ok(path)
}

/// Hide the console window that Windows would open for a console program started from a GUI program.
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

/// Start the worker with piped stdin/stdout/stderr.
pub fn spawn_worker(settings: &Settings) -> Result<(Child, PathBuf), String> {
    let python = settings::python(settings)?;
    if !python.exists() {
        return Err(format!(
            "Python was not found at {}. Run \"Set up environment\" in Settings (or `idol-studio-cli setup`), \
             or choose a Python that has Idol installed.",
            python.display()
        ));
    }
    let script = script_path()?;
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
    let child = cmd.spawn().map_err(|e| format!("could not start {}: {e}", python.display()))?;
    kill_with_parent(&child);
    Ok((child, python))
}

/// Put the child in a job object that is closed (killing the child) when this process exits, even if it is
/// killed. Otherwise a killed CLI or a crashed app would leave Python holding GPU memory.
#[cfg(windows)]
pub fn kill_with_parent(child: &Child) {
    use std::os::windows::io::AsRawHandle;
    use std::sync::OnceLock;
    use windows_sys::Win32::System::JobObjects::*;

    struct Job(isize);
    static JOB: OnceLock<Job> = OnceLock::new();
    let job = JOB.get_or_init(|| unsafe {
        let job = CreateJobObjectW(std::ptr::null(), std::ptr::null());
        if !job.is_null() {
            let mut info: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = std::mem::zeroed();
            info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
            SetInformationJobObject(
                job,
                JobObjectExtendedLimitInformation,
                &info as *const _ as *const _,
                std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
            );
        }
        Job(job as isize) // the handle is intentionally never closed: it lives as long as this process
    });
    if job.0 != 0 {
        unsafe {
            AssignProcessToJobObject(job.0 as _, child.as_raw_handle() as _);
        }
    }
}

#[cfg(not(windows))]
pub fn kill_with_parent(_child: &Child) {}
