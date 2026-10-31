pub mod cli;
mod library;
mod pathenv;
mod paths;
mod process;
mod settings;
mod setup;
mod updater;
mod worker;

use serde::Serialize;
use settings::Settings;
use std::path::PathBuf;
use tauri::{AppHandle, Manager, RunEvent};

const IMAGE_EXTENSIONS: [&str; 5] = ["png", "jpg", "jpeg", "webp", "bmp"];

#[derive(Serialize)]
struct AppInfo {
    version: String,
    repo: String,
    env_dir: String,
    managed_python: String,
    python: String,
    python_exists: bool,
    output_dir: String,
    worker_script: String,
    /// The console CLI next to the app (installed builds), or the app exe itself, which takes the same commands.
    cli_path: String,
    cli_installed: bool,
}

#[tauri::command]
fn app_info(app: AppHandle) -> Result<AppInfo, String> {
    let settings = settings::load();
    let env = settings::env_dir()?;
    let python = settings::python(&settings)?;
    Ok(AppInfo {
        version: app.package_info().version.to_string(),
        repo: updater::REPO.into(),
        managed_python: settings::env_python(&env).display().to_string(),
        env_dir: env.display().to_string(),
        python_exists: python.exists(),
        python: python.display().to_string(),
        output_dir: settings::output_dir(&settings)?.display().to_string(),
        worker_script: process::script_path().map(|p| p.display().to_string()).unwrap_or_default(),
        cli_installed: cli_path().1,
        cli_path: cli_path().0,
    })
}

fn cli_path() -> (String, bool) {
    let exe = std::env::current_exe().unwrap_or_default();
    let cli = exe.with_file_name(if cfg!(windows) { "idol-studio-cli.exe" } else { "idol-studio-cli" });
    if cli.exists() {
        (cli.display().to_string(), true)
    } else {
        (exe.display().to_string(), false)
    }
}

#[tauri::command]
fn get_settings() -> Settings {
    settings::load()
}

/// Save settings. The worker is restarted if something it was started with changed.
#[tauri::command]
fn save_settings(app: AppHandle, new_settings: Settings) -> Result<(), String> {
    let old = settings::load();
    settings::save(&new_settings)?;
    if old.python_path != new_settings.python_path || old.hf_token != new_settings.hf_token {
        if let Ok(mut worker) = app.state::<worker::WorkerState>().0.lock() {
            worker.stop();
        }
    }
    Ok(())
}

/// Raw file contents (for the splat viewer), sent as binary instead of JSON.
#[tauri::command]
fn read_file_bytes(path: String) -> Result<tauri::ipc::Response, String> {
    std::fs::read(&path)
        .map(tauri::ipc::Response::new)
        .map_err(|e| format!("{path}: {e}"))
}

#[tauri::command]
fn read_text_file(path: String) -> Result<String, String> {
    std::fs::read_to_string(&path).map_err(|e| format!("{path}: {e}"))
}

#[tauri::command]
fn path_exists(path: String) -> bool {
    PathBuf::from(path).exists()
}

/// Images in a folder (not recursive), sorted by name.
#[tauri::command]
fn list_images(dir: String) -> Result<Vec<String>, String> {
    let mut images: Vec<String> = std::fs::read_dir(&dir)
        .map_err(|e| format!("{dir}: {e}"))?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.is_file()
                && path
                    .extension()
                    .and_then(|e| e.to_str())
                    .is_some_and(|e| IMAGE_EXTENSIONS.contains(&e.to_lowercase().as_str()))
        })
        .map(|path| path.display().to_string())
        .collect();
    images.sort();
    Ok(images)
}

#[tauri::command]
fn is_dir(path: String) -> bool {
    PathBuf::from(path).is_dir()
}

/// Delete a generated file set (`<base>.ply`, `<base>_input.png`, `<base>_camera.json`).
#[tauri::command]
fn delete_outputs(paths: Vec<String>) -> Result<(), String> {
    for path in paths {
        let path = PathBuf::from(path);
        if path.is_file() {
            std::fs::remove_file(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        }
    }
    Ok(())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let app = tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_opener::init())
        .manage(worker::WorkerState::default())
        .setup(|app| {
            updater::spawn_periodic_checks(app.handle().clone());
            library::watch(app.handle().clone());
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            app_info,
            get_settings,
            save_settings,
            read_file_bytes,
            read_text_file,
            path_exists,
            list_images,
            is_dir,
            delete_outputs,
            library::library_load,
            library::library_save,
            library::library_import,
            library::library_add_export,
            pathenv::cli_path_status,
            pathenv::cli_path_set,
            worker::worker_send,
            worker::worker_stop,
            worker::worker_running,
            setup::setup_environment,
            updater::check_for_updates,
            updater::install_update,
        ])
        .build(tauri::generate_context!())
        .expect("error while building Idol Studio");

    app.run(|handle, event| {
        if let RunEvent::Exit = event {
            // Don't leave a Python process holding GPU memory behind.
            if let Ok(mut worker) = handle.state::<worker::WorkerState>().0.lock() {
                worker.stop();
            }
        }
    });
}
