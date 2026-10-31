use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use tauri::{AppHandle, Manager};

/// User settings, stored as JSON in the app config folder.
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(default)]
pub struct Settings {
    /// Python interpreter to run Idol with. Empty: the managed environment created by "Set up environment".
    pub python_path: String,
    /// Where generated splats go. Empty: Documents/Idol Studio.
    pub output_dir: String,
    /// Hugging Face token passed to the worker as HF_TOKEN. Empty: use the token from `hf auth login` / env.
    pub hf_token: String,
    /// Hugging Face repo or local folder with the Idol weights.
    pub weights: String,
    /// DINOv3 encoder precision: "bf16" or "fp32".
    pub dinov3_dtype: String,
    /// torch.compile + batched guidance (faster after the first run).
    pub optimize: bool,
    /// Check GitHub for new releases on start and every 30 minutes.
    pub check_updates: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            python_path: String::new(),
            output_dir: String::new(),
            hf_token: String::new(),
            weights: "speridlabs/idol".into(),
            dinov3_dtype: "bf16".into(),
            optimize: true,
            check_updates: true,
        }
    }
}

fn settings_file(app: &AppHandle) -> Result<PathBuf, String> {
    let dir = app.path().app_config_dir().map_err(|e| e.to_string())?;
    Ok(dir.join("settings.json"))
}

pub fn load(app: &AppHandle) -> Settings {
    settings_file(app)
        .ok()
        .and_then(|path| std::fs::read_to_string(path).ok())
        .and_then(|text| serde_json::from_str(&text).ok())
        .unwrap_or_default()
}

pub fn save(app: &AppHandle, settings: &Settings) -> Result<(), String> {
    let path = settings_file(app)?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    let text = serde_json::to_string_pretty(settings).map_err(|e| e.to_string())?;
    std::fs::write(path, text).map_err(|e| e.to_string())
}

/// The virtual environment managed by the app.
pub fn env_dir(app: &AppHandle) -> Result<PathBuf, String> {
    Ok(app.path().app_local_data_dir().map_err(|e| e.to_string())?.join("env"))
}

pub fn env_python(env: &std::path::Path) -> PathBuf {
    if cfg!(windows) {
        env.join("Scripts").join("python.exe")
    } else {
        env.join("bin").join("python")
    }
}

/// The interpreter the worker runs with.
pub fn python(app: &AppHandle, settings: &Settings) -> Result<PathBuf, String> {
    if !settings.python_path.trim().is_empty() {
        return Ok(PathBuf::from(settings.python_path.trim()));
    }
    Ok(env_python(&env_dir(app)?))
}

pub fn output_dir(app: &AppHandle, settings: &Settings) -> Result<PathBuf, String> {
    if !settings.output_dir.trim().is_empty() {
        return Ok(PathBuf::from(settings.output_dir.trim()));
    }
    let docs = app.path().document_dir().map_err(|e| e.to_string())?;
    Ok(docs.join("Idol Studio"))
}
