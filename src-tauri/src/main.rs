// Hide the console window in release builds on Windows.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    // `idol-studio.exe generate ...` etc. run the command line interface instead of the app.
    if idol_studio_lib::cli::is_cli_invocation() {
        idol_studio_lib::cli::attach_console();
        std::process::exit(idol_studio_lib::cli::run());
    }
    idol_studio_lib::run()
}
