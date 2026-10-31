//! Adding the CLI's folder to the user's PATH (HKCU\Environment\Path), so `idol-studio-cli` works in any new
//! terminal. Used by `idol-studio-cli path add|remove|status`, the installer and Settings.

use std::path::{Path, PathBuf};

/// The folder that holds idol-studio-cli.exe: next to the running exe.
pub fn cli_dir() -> Result<PathBuf, String> {
    let exe = std::env::current_exe().map_err(|e| e.to_string())?;
    exe.parent().map(Path::to_path_buf).ok_or_else(|| "no folder for the running exe".into())
}

fn same_dir(a: &str, b: &Path) -> bool {
    let norm = |s: &str| s.trim().trim_end_matches(['\\', '/']).to_lowercase();
    norm(a) == norm(&b.display().to_string())
}

#[cfg(windows)]
mod imp {
    use super::*;
    use winreg::enums::{RegType, HKEY_CURRENT_USER, KEY_READ, KEY_WRITE};
    use winreg::{RegKey, RegValue};

    fn env_key() -> Result<RegKey, String> {
        RegKey::predef(HKEY_CURRENT_USER)
            .open_subkey_with_flags("Environment", KEY_READ | KEY_WRITE)
            .map_err(|e| format!("could not open HKCU\\Environment: {e}"))
    }

    /// The user PATH as written (unexpanded) and its registry type.
    fn read(key: &RegKey) -> (String, RegType) {
        match key.get_raw_value("Path") {
            Ok(value) => {
                let units: Vec<u16> = value.bytes.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
                let text = String::from_utf16_lossy(&units).trim_end_matches('\0').to_string();
                (text, value.vtype)
            }
            Err(_) => (String::new(), RegType::REG_EXPAND_SZ),
        }
    }

    fn write(key: &RegKey, path: &str, vtype: RegType) -> Result<(), String> {
        let mut bytes: Vec<u8> = path.encode_utf16().chain(std::iter::once(0)).flat_map(u16::to_le_bytes).collect();
        if bytes.is_empty() {
            bytes = vec![0, 0];
        }
        key.set_raw_value("Path", &RegValue { bytes, vtype })
            .map_err(|e| format!("could not write the user PATH: {e}"))?;
        broadcast();
        Ok(())
    }

    /// Tell Explorer (and new terminals it starts) that the environment changed.
    fn broadcast() {
        use windows_sys::Win32::UI::WindowsAndMessaging::{SendMessageTimeoutW, HWND_BROADCAST, SMTO_ABORTIFHUNG, WM_SETTINGCHANGE};
        let name: Vec<u16> = "Environment".encode_utf16().chain(std::iter::once(0)).collect();
        let mut result = 0usize;
        unsafe {
            SendMessageTimeoutW(HWND_BROADCAST, WM_SETTINGCHANGE, 0, name.as_ptr() as isize, SMTO_ABORTIFHUNG, 3000, &mut result);
        }
    }

    pub fn contains(dir: &Path) -> Result<bool, String> {
        let (path, _) = read(&env_key()?);
        Ok(path.split(';').any(|p| same_dir(p, dir)))
    }

    /// Returns true if PATH was changed.
    pub fn add(dir: &Path) -> Result<bool, String> {
        let key = env_key()?;
        let (path, vtype) = read(&key);
        if path.split(';').any(|p| same_dir(p, dir)) {
            return Ok(false);
        }
        // Append without touching anything else; keep a trailing ';' if there was one, so remove() restores
        // the exact original value.
        let dir = dir.display().to_string();
        let new_path = if path.is_empty() {
            dir
        } else if let Some(stripped) = path.strip_suffix(';') {
            format!("{stripped};{dir};")
        } else {
            format!("{path};{dir}")
        };
        let vtype = if vtype == RegType::REG_SZ { RegType::REG_SZ } else { RegType::REG_EXPAND_SZ };
        write(&key, &new_path, vtype)?;
        Ok(true)
    }

    /// Returns true if PATH was changed.
    pub fn remove(dir: &Path) -> Result<bool, String> {
        let key = env_key()?;
        let (path, vtype) = read(&key);
        // Drop only our entry; empty entries and order stay as they were.
        let parts: Vec<&str> = path.split(';').collect();
        let kept: Vec<&str> = parts.iter().copied().filter(|p| p.trim().is_empty() || !same_dir(p, dir)).collect();
        if kept.len() == parts.len() {
            return Ok(false);
        }
        write(&key, &kept.join(";"), vtype)?;
        Ok(true)
    }
}

#[cfg(not(windows))]
mod imp {
    use super::*;
    pub fn contains(_dir: &Path) -> Result<bool, String> {
        Ok(false)
    }
    pub fn add(_dir: &Path) -> Result<bool, String> {
        Err("adding to PATH is only supported on Windows".into())
    }
    pub fn remove(_dir: &Path) -> Result<bool, String> {
        Err("removing from PATH is only supported on Windows".into())
    }
}

pub use imp::{add, contains, remove};

#[derive(serde::Serialize)]
pub struct PathStatus {
    pub dir: String,
    pub on_path: bool,
    pub cli_exists: bool,
}

pub fn status() -> Result<PathStatus, String> {
    let dir = cli_dir()?;
    Ok(PathStatus {
        on_path: contains(&dir)?,
        cli_exists: dir.join(if cfg!(windows) { "idol-studio-cli.exe" } else { "idol-studio-cli" }).exists(),
        dir: dir.display().to_string(),
    })
}

#[tauri::command]
pub fn cli_path_status() -> Result<PathStatus, String> {
    status()
}

#[tauri::command]
pub fn cli_path_set(on: bool) -> Result<PathStatus, String> {
    let dir = cli_dir()?;
    if on {
        add(&dir)?;
    } else {
        remove(&dir)?;
    }
    status()
}
