//! Update check against the GitHub releases of this repo.
//!
//! Checks shortly after start, then every 30 minutes (if enabled in Settings), and on demand. When a newer
//! release exists the UI gets `update-available`; "Install" downloads the release's setup .exe and runs it.

use crate::settings;
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use std::time::Duration;
use tauri::{AppHandle, Emitter};
use tokio::io::AsyncWriteExt;

pub const REPO: &str = "soundtrackgeek/idol";
const CHECK_EVERY: Duration = Duration::from_secs(30 * 60);
const FIRST_CHECK_AFTER: Duration = Duration::from_secs(5);

#[derive(Serialize, Clone, Debug)]
pub struct UpdateInfo {
    pub current: String,
    pub latest: Option<String>,
    pub available: bool,
    pub notes: String,
    pub page_url: String,
    pub asset_url: Option<String>,
    pub asset_name: Option<String>,
    pub checked_at: u64,
}

#[derive(Deserialize)]
struct Release {
    tag_name: String,
    html_url: String,
    #[serde(default)]
    body: Option<String>,
    #[serde(default)]
    draft: bool,
    #[serde(default)]
    prerelease: bool,
    #[serde(default)]
    assets: Vec<Asset>,
}

#[derive(Deserialize)]
struct Asset {
    name: String,
    browser_download_url: String,
}

fn client() -> Result<reqwest::Client, String> {
    reqwest::Client::builder()
        .user_agent(concat!("IdolStudio/", env!("CARGO_PKG_VERSION")))
        .timeout(Duration::from_secs(30))
        .build()
        .map_err(|e| e.to_string())
}

fn now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn parse_version(tag: &str) -> Option<semver::Version> {
    semver::Version::parse(tag.trim().trim_start_matches(['v', 'V'])).ok()
}

pub async fn fetch(current: &str) -> Result<UpdateInfo, String> {
    let url = format!("https://api.github.com/repos/{REPO}/releases/latest");
    let response = client()?
        .get(url)
        .header("Accept", "application/vnd.github+json")
        .send()
        .await
        .map_err(|e| format!("could not reach GitHub: {e}"))?;
    let mut info = UpdateInfo {
        current: current.to_string(),
        latest: None,
        available: false,
        notes: String::new(),
        page_url: format!("https://github.com/{REPO}/releases"),
        asset_url: None,
        asset_name: None,
        checked_at: now(),
    };
    if response.status() == reqwest::StatusCode::NOT_FOUND {
        return Ok(info); // no releases published yet
    }
    if !response.status().is_success() {
        return Err(format!("GitHub answered {}", response.status()));
    }
    let release: Release = response.json().await.map_err(|e| e.to_string())?;
    if release.draft || release.prerelease {
        return Ok(info);
    }
    let latest = parse_version(&release.tag_name);
    let current_version = parse_version(current);
    info.latest = Some(release.tag_name.trim_start_matches(['v', 'V']).to_string());
    info.available = matches!((&latest, &current_version), (Some(l), Some(c)) if l > c);
    info.notes = release.body.unwrap_or_default();
    info.page_url = release.html_url;
    // Prefer the installer; fall back to any .exe.
    let asset = release
        .assets
        .iter()
        .find(|a| a.name.to_lowercase().ends_with("-setup.exe"))
        .or_else(|| release.assets.iter().find(|a| a.name.to_lowercase().ends_with(".exe")));
    if let Some(asset) = asset {
        info.asset_url = Some(asset.browser_download_url.clone());
        info.asset_name = Some(asset.name.clone());
    }
    Ok(info)
}

fn current_version(app: &AppHandle) -> String {
    app.package_info().version.to_string()
}

#[tauri::command]
pub async fn check_for_updates(app: AppHandle) -> Result<UpdateInfo, String> {
    let info = fetch(&current_version(&app)).await?;
    let _ = app.emit("update-checked", info.clone());
    if info.available {
        let _ = app.emit("update-available", info.clone());
    }
    Ok(info)
}

/// Background checks: once shortly after start, then every 30 minutes.
pub fn spawn_periodic_checks(app: AppHandle) {
    tauri::async_runtime::spawn(async move {
        tokio::time::sleep(FIRST_CHECK_AFTER).await;
        loop {
            if settings::load().check_updates {
                match fetch(&current_version(&app)).await {
                    Ok(info) => {
                        let _ = app.emit("update-checked", info.clone());
                        if info.available {
                            let _ = app.emit("update-available", info);
                        }
                    }
                    Err(error) => {
                        let _ = app.emit("update-error", error);
                    }
                }
            }
            tokio::time::sleep(CHECK_EVERY).await;
        }
    });
}

#[derive(Serialize, Clone)]
struct DownloadProgress {
    received: u64,
    total: Option<u64>,
}

/// Download the installer to the temp folder, start it and quit so it can replace the app.
#[tauri::command]
pub async fn install_update(app: AppHandle, url: String, name: String) -> Result<(), String> {
    if !url.starts_with(&format!("https://github.com/{REPO}/releases/download/")) {
        return Err("refusing to download an update from outside this app's GitHub releases".into());
    }
    let safe_name: String = name
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || "._-".contains(c) { c } else { '_' })
        .collect();
    let path = std::env::temp_dir().join(if safe_name.is_empty() { "IdolStudio-setup.exe".into() } else { safe_name });

    let response = client()?
        .get(&url)
        .timeout(Duration::from_secs(600))
        .send()
        .await
        .map_err(|e| e.to_string())?
        .error_for_status()
        .map_err(|e| e.to_string())?;
    let total = response.content_length();
    let mut file = tokio::fs::File::create(&path).await.map_err(|e| e.to_string())?;
    let mut received = 0u64;
    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| e.to_string())?;
        file.write_all(&chunk).await.map_err(|e| e.to_string())?;
        received += chunk.len() as u64;
        let _ = app.emit("update-progress", DownloadProgress { received, total });
    }
    file.flush().await.map_err(|e| e.to_string())?;
    drop(file);

    // Stop the GPU worker before the installer replaces files.
    if let Ok(mut worker) = tauri::Manager::state::<crate::worker::WorkerState>(&app).0.lock() {
        worker.stop();
    }
    std::process::Command::new(&path)
        .spawn()
        .map_err(|e| format!("could not start the installer: {e}"))?;
    app.exit(0);
    Ok(())
}
