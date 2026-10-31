//! The image library (images and their results), stored in library.json in the app config folder.
//!
//! Shared by the GUI and the CLI: results generated from the command line show up in the app. Changes are
//! read-modify-write under a lock file, and the GUI is told when another process changed the file.

use crate::paths;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::{Duration, SystemTime};
use tauri::{AppHandle, Emitter};

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct GenParams {
    pub mode: String,
    pub crop_margin: Option<f64>,
    pub seed: u64,
    pub steps: u32,
    pub cfg_strength: f64,
    pub rescale_t: f64,
    pub num_gaussians: u64,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct LibResult {
    pub id: String,
    pub ply: String,
    pub preprocessed: String,
    pub camera_path: String,
    pub params: GenParams,
    pub seconds: f64,
    pub num_gaussians: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vram_peak_gb: Option<f64>,
    /// Milliseconds since 1970 (JavaScript Date.now()).
    pub created: u64,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct LibraryImage {
    pub id: String,
    pub path: String,
    pub name: String,
    #[serde(default)]
    pub results: Vec<LibResult>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub preview: Option<String>,
}

pub fn file() -> Result<PathBuf, String> {
    Ok(paths::config_dir()?.join("library.json"))
}

pub fn now_ms() -> u64 {
    SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).map(|d| d.as_millis() as u64).unwrap_or(0)
}

pub fn new_id() -> String {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let nanos = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0);
    format!("{:x}{:x}{:x}", nanos as u64 & 0xffff_ffff_ffff, std::process::id(), COUNTER.fetch_add(1, Ordering::Relaxed))
}

pub fn load() -> Vec<LibraryImage> {
    file()
        .ok()
        .and_then(|path| std::fs::read_to_string(path).ok())
        .and_then(|text| serde_json::from_str(&text).ok())
        .unwrap_or_default()
}

fn save_unlocked(library: &[LibraryImage]) -> Result<(), String> {
    let text = serde_json::to_string_pretty(library).map_err(|e| e.to_string())?;
    let path = file()?;
    paths::write_atomic(&path, text.as_bytes())?;
    remember_own_write(&path);
    Ok(())
}

/// Hold a lock file while `f` runs. A lock older than 30 s is treated as left over from a crash.
fn with_lock<T>(f: impl FnOnce() -> Result<T, String>) -> Result<T, String> {
    let lock = file()?.with_extension("lock");
    if let Some(parent) = lock.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    let started = std::time::Instant::now();
    loop {
        match std::fs::OpenOptions::new().write(true).create_new(true).open(&lock) {
            Ok(_) => break,
            Err(_) => {
                let stale = std::fs::metadata(&lock)
                    .and_then(|m| m.modified())
                    .map(|t| t.elapsed().unwrap_or_default() > Duration::from_secs(30))
                    .unwrap_or(true);
                if stale || started.elapsed() > Duration::from_secs(10) {
                    let _ = std::fs::remove_file(&lock);
                } else {
                    std::thread::sleep(Duration::from_millis(25));
                }
            }
        }
    }
    let result = f();
    let _ = std::fs::remove_file(&lock);
    result
}

/// Load, change and save the library as one step.
pub fn update<T>(change: impl FnOnce(&mut Vec<LibraryImage>) -> T) -> Result<T, String> {
    with_lock(|| {
        let mut library = load();
        let out = change(&mut library);
        save_unlocked(&library)?;
        Ok(out)
    })
}

/// Find the image with this path, adding it if needed. Returns its index.
pub fn ensure_image(library: &mut Vec<LibraryImage>, path: &Path) -> usize {
    let path_str = path.display().to_string();
    if let Some(index) = library.iter().position(|i| i.path.eq_ignore_ascii_case(&path_str)) {
        return index;
    }
    let name = path.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_else(|| path_str.clone());
    library.push(LibraryImage { id: new_id(), path: path_str, name, results: vec![], preview: None });
    library.len() - 1
}

// ---------------------------------------------------------------------------------------------------------------
// Change notification for the GUI

static OWN_WRITE: Mutex<Option<SystemTime>> = Mutex::new(None);

fn modified(path: &Path) -> Option<SystemTime> {
    std::fs::metadata(path).and_then(|m| m.modified()).ok()
}

fn remember_own_write(path: &Path) {
    if let Ok(mut own) = OWN_WRITE.lock() {
        *own = modified(path);
    }
}

/// Emit `library-changed` when another process (the CLI) changes library.json.
pub fn watch(app: AppHandle) {
    std::thread::spawn(move || {
        let Ok(path) = file() else { return };
        let mut seen = modified(&path);
        loop {
            std::thread::sleep(Duration::from_millis(1000));
            let now = modified(&path);
            if now != seen {
                seen = now;
                let own = OWN_WRITE.lock().ok().and_then(|o| *o);
                if now != own {
                    let _ = app.emit("library-changed", ());
                }
            }
        }
    });
}

// ---------------------------------------------------------------------------------------------------------------
// GUI commands

#[tauri::command]
pub fn library_load() -> Vec<LibraryImage> {
    load()
}

#[tauri::command]
pub fn library_save(library: Vec<LibraryImage>) -> Result<(), String> {
    with_lock(|| save_unlocked(&library))
}

/// First run after the library moved out of the webview: merge the old localStorage copy into the file (the CLI
/// may already have created it). Images match by path, results by .ply. Returns how many results were added.
#[tauri::command]
pub fn library_import(library: Vec<LibraryImage>) -> Result<usize, String> {
    update(|current| {
        let mut added = 0;
        for old in library {
            match current.iter_mut().find(|i| i.path.eq_ignore_ascii_case(&old.path)) {
                Some(image) => {
                    for result in old.results {
                        if !image.results.iter().any(|r| r.ply.eq_ignore_ascii_case(&result.ply)) {
                            image.results.push(result);
                            added += 1;
                        }
                    }
                    image.results.sort_by(|a, b| b.created.cmp(&a.created));
                    if image.preview.is_none() {
                        image.preview = old.preview;
                    }
                }
                None => {
                    added += old.results.len();
                    current.push(old);
                }
            }
        }
        added
    })
}
