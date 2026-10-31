//! Console build of the command line interface. Same commands as `idol-studio.exe <command>`, but as a console
//! program, so terminals wait for it and show its output.

fn main() {
    std::process::exit(idol_studio_lib::cli::run());
}
