//! App folders, resolved without a Tauri handle so the CLI and the GUI agree.
//! Same locations Tauri uses: %APPDATA%\<id> (config), %LOCALAPPDATA%\<id> (data), the Documents folder.

use std::path::PathBuf;

pub const IDENTIFIER: &str = "com.soundtrackgeek.idolstudio";

pub fn config_dir() -> Result<PathBuf, String> {
    dirs::config_dir().map(|d| d.join(IDENTIFIER)).ok_or_else(|| "no config folder".into())
}

pub fn local_data_dir() -> Result<PathBuf, String> {
    dirs::data_local_dir().map(|d| d.join(IDENTIFIER)).ok_or_else(|| "no local data folder".into())
}

pub fn documents_dir() -> Result<PathBuf, String> {
    dirs::document_dir().or_else(dirs::home_dir).ok_or_else(|| "no Documents folder".into())
}

/// Write a file atomically (temp file + rename), so readers never see half a file.
pub fn write_atomic(path: &std::path::Path, contents: &[u8]) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    let tmp = path.with_extension(format!("tmp{}", std::process::id()));
    std::fs::write(&tmp, contents).map_err(|e| format!("{}: {e}", tmp.display()))?;
    std::fs::rename(&tmp, path).map_err(|e| format!("{}: {e}", path.display()))
}
