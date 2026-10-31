//! Command line interface: `idol-studio-cli <command>` (or `idol-studio.exe <command>`).
//!
//! Every command can print JSON (`--json`, one object per line on stdout) for scripts and agents. Generated
//! results are added to the app's library unless `--no-library` is given. `serve` keeps the model loaded so
//! later `generate` calls skip the load and compile.

use crate::library::{self, GenParams, LibResult};
use crate::{paths, process, settings, setup};
use clap::{Args, Parser, Subcommand, ValueEnum};
use serde_json::{json, Value};
use std::collections::VecDeque;
use std::io::{BufRead, BufReader, IsTerminal, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin};
use std::sync::mpsc::{self, Receiver};
use std::sync::{Arc, Mutex};
use std::time::Duration;

pub const EXIT_OK: i32 = 0;
pub const EXIT_ERROR: i32 = 1;
pub const EXIT_USAGE: i32 = 2;
pub const EXIT_ACCESS: i32 = 3;
pub const DEFAULT_PORT: u16 = 47817;
const IMAGE_EXTENSIONS: [&str; 5] = ["png", "jpg", "jpeg", "webp", "bmp"];

const SUBCOMMANDS: [&str; 8] = ["generate", "preview", "check-access", "info", "setup", "serve", "config", "library"];

/// True when the process was started with a CLI command (or --help / --version) rather than as the app.
pub fn is_cli_invocation() -> bool {
    std::env::args()
        .nth(1)
        .is_some_and(|a| SUBCOMMANDS.contains(&a.as_str()) || ["help", "--help", "-h", "--version", "-V"].contains(&a.as_str()))
}

// ---------------------------------------------------------------------------------------------------------------
// Arguments

#[derive(Parser)]
#[command(
    name = "idol-studio-cli",
    version,
    about = "Idol Studio command line: turn single images into 3D Gaussian splats (.ply) with Idol.",
    long_about = "Idol Studio command line: turn single images into 3D Gaussian splats (.ply) with Idol.\n\n\
        Uses the same settings, Python environment, Hugging Face token, output folder and image library as the app.\n\
        Add --json to any command for machine-readable output (one JSON object per line on stdout).\n\n\
        Exit codes: 0 ok, 1 error, 2 bad arguments, 3 Hugging Face access missing.",
    after_help = "Examples:\n  \
        idol-studio-cli generate photo.jpg --mode real\n  \
        idol-studio-cli generate renders/ --variants 3 --gaussians 524288 --json\n  \
        idol-studio-cli serve            (in another terminal: keeps the model loaded)\n  \
        idol-studio-cli check-access --json"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Generate splats from images or folders of images
    Generate(GenerateArgs),
    /// Save the background-removed, cropped image the model would see (fast: loads only BiRefNet)
    Preview(PreviewArgs),
    /// Check Hugging Face access to the gated encoders Idol needs
    CheckAccess(PipelineArgs),
    /// Show versions, paths, settings and the GPU
    Info(InfoArgs),
    /// Create or repair the Python environment (CUDA PyTorch, Idol, Triton)
    Setup(JsonArg),
    /// Keep a worker with the model loaded; `generate` and `preview` use it automatically
    Serve(ServeArgs),
    /// Show or change settings
    Config(ConfigArgs),
    /// List the images and results in the app's library
    Library(JsonArg),
}

#[derive(Args, Clone, Copy)]
struct JsonArg {
    /// Print JSON (one object per line) on stdout
    #[arg(long)]
    json: bool,
}

#[derive(Clone, Copy, ValueEnum)]
enum Mode {
    /// Renders and generated images of one object (crop 1.2)
    Synthetic,
    /// Real-world photos (crop 1.7)
    Real,
}

impl Mode {
    fn as_str(self) -> &'static str {
        match self {
            Mode::Synthetic => "synthetic",
            Mode::Real => "real",
        }
    }
}

#[derive(Clone, Copy, ValueEnum)]
enum Dtype {
    Bf16,
    Fp32,
}

#[derive(Args, Clone)]
struct PipelineArgs {
    /// Hugging Face repo or local folder with the Idol weights [default: from settings]
    #[arg(long)]
    weights: Option<String>,
    /// DINOv3 encoder precision; fp32 uses 1.7 GB more VRAM [default: from settings]
    #[arg(long, value_enum)]
    dtype: Option<Dtype>,
    /// Use torch.compile for faster sampling (the first run compiles)
    #[arg(long, overrides_with = "no_optimize")]
    optimize: bool,
    /// Run the sampler without torch.compile
    #[arg(long, overrides_with = "optimize")]
    no_optimize: bool,
    /// Don't use a running `serve` instance; start a worker just for this command
    #[arg(long)]
    no_server: bool,
    /// Show the Python worker's log on stderr
    #[arg(long, short)]
    verbose: bool,
    /// Print JSON (one object per line) on stdout
    #[arg(long)]
    json: bool,
}

#[derive(Args)]
struct GenerateArgs {
    /// Image files and/or folders (png, jpg, jpeg, webp, bmp)
    #[arg(required = true)]
    images: Vec<PathBuf>,
    /// Output folder; each image gets a subfolder [default: from settings, Documents\Idol Studio]
    #[arg(long, short)]
    out: Option<PathBuf>,
    /// Input type
    #[arg(long, value_enum, default_value = "synthetic")]
    mode: Mode,
    /// Crop margin: square crop side = object bounding box x this [default: 1.2 synthetic, 1.7 real]
    #[arg(long)]
    crop: Option<f64>,
    /// Seed; variants use seed, seed+1, ...
    #[arg(long, default_value_t = 42)]
    seed: u64,
    /// Use a random seed for every run instead of --seed
    #[arg(long)]
    random_seed: bool,
    /// Runs per image
    #[arg(long, default_value_t = 1, value_parser = clap::value_parser!(u32).range(1..=64))]
    variants: u32,
    /// Sampling steps
    #[arg(long, default_value_t = 50, value_parser = clap::value_parser!(u32).range(1..=500))]
    steps: u32,
    /// Classifier-free guidance strength
    #[arg(long, default_value_t = 3.0)]
    cfg: f64,
    /// Time shift: higher spends more steps at high noise
    #[arg(long, default_value_t = 3.0)]
    shift: f64,
    /// Number of Gaussians: 32768 to 1048576, a multiple of 32
    #[arg(long, default_value_t = 262_144)]
    gaussians: u64,
    /// Don't add the results to the app's library
    #[arg(long)]
    no_library: bool,
    #[command(flatten)]
    pipeline: PipelineArgs,
}

#[derive(Args)]
struct PreviewArgs {
    /// Image file
    image: PathBuf,
    /// Output .png [default: <output folder>\<name>\<name>_preview_<mode>.png]
    #[arg(long, short)]
    out: Option<PathBuf>,
    #[arg(long, value_enum, default_value = "synthetic")]
    mode: Mode,
    /// Crop margin override
    #[arg(long)]
    crop: Option<f64>,
    #[command(flatten)]
    pipeline: PipelineArgs,
}

#[derive(Args)]
struct InfoArgs {
    /// Don't start Python to report PyTorch, CUDA and the GPU
    #[arg(long)]
    no_worker: bool,
    #[arg(long)]
    json: bool,
}

#[derive(Args)]
struct ServeArgs {
    /// Port on 127.0.0.1
    #[arg(long, default_value_t = DEFAULT_PORT)]
    port: u16,
    /// Don't load the model until the first request
    #[arg(long)]
    lazy: bool,
    #[command(flatten)]
    pipeline: PipelineArgs,
}

#[derive(Args)]
struct ConfigArgs {
    #[command(subcommand)]
    action: Option<ConfigAction>,
    /// Show the Hugging Face token in full
    #[arg(long)]
    show_token: bool,
    #[arg(long)]
    json: bool,
}

#[derive(Subcommand)]
enum ConfigAction {
    /// Print one setting
    Get { key: String },
    /// Change a setting: python_path, output_dir, hf_token, weights, dinov3_dtype, optimize, check_updates
    Set { key: String, value: String },
}

// ---------------------------------------------------------------------------------------------------------------
// Output

struct Out {
    json: bool,
}

impl Out {
    fn event(&self, value: Value) {
        if self.json {
            let mut stdout = std::io::stdout().lock();
            let _ = writeln!(stdout, "{value}");
            let _ = stdout.flush();
        }
    }
    fn say(&self, text: impl AsRef<str>) {
        if !self.json {
            eprintln!("{}", text.as_ref());
        }
    }
    fn error(&self, message: &str) {
        if self.json {
            self.event(json!({"event": "error", "message": message}));
        } else {
            eprintln!("error: {message}");
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Talking to a worker: a local child process, or a `serve` instance over TCP

/// Which event finishes a request.
fn answer_for(cmd: &str) -> &'static str {
    match cmd {
        "generate" => "done",
        "preprocess" => "preprocessed",
        "check_access" => "access",
        "hello" => "hello",
        "load" => "loaded",
        _ => "done",
    }
}

fn is_terminal(event: &Value, id: &str, answer: &str) -> bool {
    let kind = event["event"].as_str().unwrap_or("");
    event["id"].as_str() == Some(id) && (kind == answer || kind == "error" || kind == "cancelled")
}

struct LocalWorker {
    child: Child,
    stdin: Arc<Mutex<ChildStdin>>,
    events: Receiver<Value>,
    log_tail: Arc<Mutex<VecDeque<String>>>,
}

impl LocalWorker {
    fn start(verbose: bool) -> Result<Self, String> {
        let (mut child, _) = process::spawn_worker(&settings::load())?;
        let stdout = child.stdout.take().ok_or("no stdout")?;
        let stderr = child.stderr.take().ok_or("no stderr")?;
        let stdin = Arc::new(Mutex::new(child.stdin.take().ok_or("no stdin")?));
        let (tx, events) = mpsc::channel();
        std::thread::spawn(move || {
            for line in BufReader::new(stdout).lines().map_while(Result::ok) {
                if let Ok(value) = serde_json::from_str::<Value>(&line) {
                    if tx.send(value).is_err() {
                        break;
                    }
                }
            }
        });
        let log_tail = Arc::new(Mutex::new(VecDeque::new()));
        let tail = log_tail.clone();
        std::thread::spawn(move || {
            for line in BufReader::new(stderr).lines().map_while(Result::ok) {
                if verbose {
                    eprintln!("  | {line}");
                }
                if let Ok(mut tail) = tail.lock() {
                    tail.push_back(line);
                    if tail.len() > 30 {
                        tail.pop_front();
                    }
                }
            }
        });
        Ok(Self { child, stdin, events, log_tail })
    }

    fn write(stdin: &Mutex<ChildStdin>, request: &Value) -> Result<(), String> {
        let mut stdin = stdin.lock().map_err(|e| e.to_string())?;
        writeln!(stdin, "{request}").and_then(|_| stdin.flush()).map_err(|e| format!("could not talk to the worker: {e}"))
    }

    fn died(&mut self) -> String {
        let code = self.child.wait().ok().and_then(|s| s.code());
        let tail: Vec<String> = self.log_tail.lock().map(|t| t.iter().cloned().collect()).unwrap_or_default();
        format!("the Python worker exited (code {code:?}). Last log lines:\n{}", tail.join("\n"))
    }

    /// Send a request and pass its events to `on_event` until the answer arrives. `on_event` returns false to
    /// cancel the request (the answer is still awaited).
    fn request(&mut self, request: &Value, on_event: &mut dyn FnMut(&Value) -> bool) -> Result<Value, String> {
        let id = request["id"].as_str().unwrap_or("").to_string();
        let answer = answer_for(request["cmd"].as_str().unwrap_or(""));
        Self::write(&self.stdin, request)?;
        let mut cancelled = false;
        loop {
            match self.events.recv_timeout(Duration::from_millis(500)) {
                Ok(event) => {
                    if is_terminal(&event, &id, answer) {
                        return Ok(event);
                    }
                    let mine = event["id"].as_str().is_none_or(|e| e == id);
                    if mine && !on_event(&event) && !cancelled {
                        cancelled = true;
                        let _ = Self::write(&self.stdin, &json!({"cmd": "cancel"}));
                    }
                }
                Err(mpsc::RecvTimeoutError::Timeout) => {
                    if let Ok(Some(_)) = self.child.try_wait() {
                        return Err(self.died());
                    }
                }
                Err(mpsc::RecvTimeoutError::Disconnected) => return Err(self.died()),
            }
        }
    }

    fn stop(&mut self) {
        let _ = Self::write(&self.stdin, &json!({"cmd": "quit"}));
        for _ in 0..30 {
            if let Ok(Some(_)) = self.child.try_wait() {
                return;
            }
            std::thread::sleep(Duration::from_millis(100));
        }
        let _ = self.child.kill();
    }
}

struct RemoteWorker {
    reader: BufReader<TcpStream>,
    writer: TcpStream,
}

impl RemoteWorker {
    fn connect() -> Option<(Self, u16)> {
        let state: Value = serde_json::from_str(&std::fs::read_to_string(serve_file().ok()?).ok()?).ok()?;
        let port = state["port"].as_u64()? as u16;
        let address = std::net::SocketAddr::from(([127, 0, 0, 1], port));
        let stream = TcpStream::connect_timeout(&address, Duration::from_millis(500)).ok()?;
        // The server answers a hello line right away; anything else is not ours.
        let mut reader = BufReader::new(stream.try_clone().ok()?);
        stream.set_read_timeout(Some(Duration::from_secs(5))).ok()?;
        let mut line = String::new();
        reader.read_line(&mut line).ok()?;
        let hello: Value = serde_json::from_str(&line).ok()?;
        if hello["event"] != "idol-studio-serve" {
            return None;
        }
        stream.set_read_timeout(None).ok()?;
        Some((Self { reader, writer: stream }, port))
    }

    fn request(&mut self, request: &Value, on_event: &mut dyn FnMut(&Value) -> bool) -> Result<Value, String> {
        let id = request["id"].as_str().unwrap_or("").to_string();
        let answer = answer_for(request["cmd"].as_str().unwrap_or(""));
        writeln!(self.writer, "{request}").map_err(|e| format!("lost the connection to the server: {e}"))?;
        let mut cancelled = false;
        loop {
            let mut line = String::new();
            let read = self.reader.read_line(&mut line).map_err(|e| format!("lost the connection to the server: {e}"))?;
            if read == 0 {
                return Err("the server closed the connection".into());
            }
            let Ok(event) = serde_json::from_str::<Value>(&line) else { continue };
            if is_terminal(&event, &id, answer) {
                return Ok(event);
            }
            if !on_event(&event) && !cancelled {
                cancelled = true;
                let _ = writeln!(self.writer, "{}", json!({"cmd": "cancel"}));
            }
        }
    }
}

enum Backend {
    Local(LocalWorker),
    Remote(RemoteWorker),
}

impl Backend {
    fn open(args: &PipelineArgs, out: &Out) -> Result<Self, String> {
        if !args.no_server {
            if let Some((remote, port)) = RemoteWorker::connect() {
                out.say(format!("Using the running server on port {port}."));
                return Ok(Backend::Remote(remote));
            }
        }
        out.say("Starting the Python worker...");
        Ok(Backend::Local(LocalWorker::start(args.verbose)?))
    }

    fn request(&mut self, request: &Value, on_event: &mut dyn FnMut(&Value) -> bool) -> Result<Value, String> {
        match self {
            Backend::Local(worker) => worker.request(request, on_event),
            Backend::Remote(worker) => worker.request(request, on_event),
        }
    }

    fn close(&mut self) {
        if let Backend::Local(worker) = self {
            worker.stop();
        }
    }
}

/// Pipeline settings for a request: command line first, then the app settings.
fn pipeline_config(args: &PipelineArgs) -> Value {
    let settings = settings::load();
    let optimize = if args.optimize {
        true
    } else if args.no_optimize {
        false
    } else {
        settings.optimize
    };
    json!({
        "weights": args.weights.clone().unwrap_or(settings.weights),
        "dinov3_dtype": match args.dtype { Some(Dtype::Fp32) => "fp32".to_string(), Some(Dtype::Bf16) => "bf16".to_string(), None => settings.dinov3_dtype },
        "optimize": optimize,
    })
}

fn merge(mut a: Value, b: &Value) -> Value {
    if let (Some(a), Some(b)) = (a.as_object_mut(), b.as_object()) {
        for (k, v) in b {
            a.insert(k.clone(), v.clone());
        }
    }
    a
}

// ---------------------------------------------------------------------------------------------------------------
// Helpers

fn absolute(path: &Path) -> PathBuf {
    std::path::absolute(path).unwrap_or_else(|_| path.to_path_buf())
}

fn is_image(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| IMAGE_EXTENSIONS.contains(&e.to_lowercase().as_str()))
}

/// Expand folders (not recursive) and check that every file is an image.
fn collect_images(inputs: &[PathBuf]) -> Result<Vec<PathBuf>, String> {
    let mut images = Vec::new();
    for input in inputs {
        let path = absolute(input);
        if path.is_dir() {
            let mut found: Vec<PathBuf> = std::fs::read_dir(&path)
                .map_err(|e| format!("{}: {e}", path.display()))?
                .filter_map(Result::ok)
                .map(|e| e.path())
                .filter(|p| p.is_file() && is_image(p))
                .collect();
            found.sort();
            if found.is_empty() {
                return Err(format!("{}: no images in this folder", path.display()));
            }
            images.extend(found);
        } else if path.is_file() {
            if !is_image(&path) {
                return Err(format!("{}: not a png, jpg, jpeg, webp or bmp image", path.display()));
            }
            images.push(path);
        } else {
            return Err(format!("{}: file not found", path.display()));
        }
    }
    Ok(images)
}

fn stem(path: &Path) -> String {
    let stem = path.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_else(|| "image".into());
    stem.chars().map(|c| if c.is_alphanumeric() || "._-".contains(c) { c } else { '_' }).collect()
}

/// UTC timestamp like 20261031-121115 (same format as the app).
fn timestamp() -> String {
    let secs = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0) as i64;
    let (days, rem) = (secs.div_euclid(86_400), secs.rem_euclid(86_400));
    // Civil date from days since 1970-01-01 (Howard Hinnant's algorithm).
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + if month <= 2 { 1 } else { 0 };
    format!("{year:04}{month:02}{day:02}-{:02}{:02}{:02}", rem / 3600, rem % 3600 / 60, rem % 60)
}

fn random_seed() -> u64 {
    use std::hash::BuildHasher;
    std::collections::hash_map::RandomState::new().hash_one(library::new_id()) % (1 << 31)
}

fn mask_token(token: &str) -> String {
    let token = token.trim();
    if token.is_empty() {
        String::new()
    } else if token.len() <= 8 {
        "****".into()
    } else {
        format!("{}…{}", &token[..3], &token[token.len() - 4..])
    }
}

fn serve_file() -> Result<PathBuf, String> {
    Ok(paths::local_data_dir()?.join("serve.json"))
}

fn access_error(message: &str) -> bool {
    message.contains("gated") || message.contains("Hugging Face")
}

// ---------------------------------------------------------------------------------------------------------------
// Commands

fn cmd_generate(args: GenerateArgs) -> i32 {
    let out = Out { json: args.pipeline.json };
    if !(32_768..=1_048_576).contains(&args.gaussians) || args.gaussians % 32 != 0 {
        out.error("--gaussians must be between 32768 and 1048576 and a multiple of 32");
        return EXIT_USAGE;
    }
    let images = match collect_images(&args.images) {
        Ok(images) => images,
        Err(e) => {
            out.error(&e);
            return EXIT_USAGE;
        }
    };
    let settings = settings::load();
    let out_dir = match args.out.clone().map(|o| absolute(&o)).map(Ok).unwrap_or_else(|| settings::output_dir(&settings)) {
        Ok(dir) => dir,
        Err(e) => {
            out.error(&e);
            return EXIT_ERROR;
        }
    };

    // One job per image and variant.
    let mut jobs = Vec::new();
    for image in &images {
        for variant in 0..args.variants {
            let seed = if args.random_seed { random_seed() } else { args.seed + variant as u64 };
            jobs.push((image.clone(), seed));
        }
    }
    let total = jobs.len();
    out.event(json!({"event": "start", "jobs": total, "images": images.len(), "out_dir": out_dir}));
    out.say(format!("{total} run{} for {} image{} -> {}", if total == 1 { "" } else { "s" }, images.len(), if images.len() == 1 { "" } else { "s" }, out_dir.display()));

    let mut backend = match Backend::open(&args.pipeline, &out) {
        Ok(b) => b,
        Err(e) => {
            out.error(&e);
            return EXIT_ERROR;
        }
    };
    let pipeline = pipeline_config(&args.pipeline);
    let terminal = std::io::stderr().is_terminal() && !out.json;
    let mut results = Vec::new();
    let mut failed = 0;
    let mut exit = EXIT_OK;

    for (index, (image, seed)) in jobs.iter().enumerate() {
        let n = index + 1;
        let name = image.file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
        let base = out_dir.join(stem(image)).join(format!(
            "{}_{}_s{}_{}k_{}",
            stem(image),
            args.mode.as_str(),
            seed,
            args.gaussians / 1024,
            timestamp()
        ));
        let params = GenParams {
            mode: args.mode.as_str().into(),
            crop_margin: args.crop,
            seed: *seed,
            steps: args.steps,
            cfg_strength: args.cfg,
            rescale_t: args.shift,
            num_gaussians: args.gaussians,
        };
        let id = format!("cli-{}-{n}", library::new_id());
        let request = merge(
            json!({
                "cmd": "generate", "id": id, "image": image, "out_base": base,
                "mode": params.mode, "crop_margin": params.crop_margin, "seed": params.seed, "steps": params.steps,
                "cfg_strength": params.cfg_strength, "rescale_t": params.rescale_t, "num_gaussians": params.num_gaussians,
            }),
            &pipeline,
        );
        out.event(json!({"event": "job", "job": n, "of": total, "image": image, "seed": seed}));
        out.say(format!("[{n}/{total}] {name} seed {seed}"));

        let mut on_event = |event: &Value| -> bool {
            match event["event"].as_str().unwrap_or("") {
                "status" => {
                    let message = event["message"].as_str().unwrap_or("");
                    out.event(json!({"event": "status", "job": n, "stage": event["stage"], "message": message}));
                    if terminal {
                        eprint!("\r\x1b[2K");
                    }
                    out.say(format!("  {message}"));
                }
                "loaded" => {
                    out.event(json!({"event": "loaded", "job": n, "seconds": event["seconds"], "vram_allocated_gb": event["vram_allocated_gb"]}));
                    out.say(format!("  model loaded in {}s", event["seconds"]));
                }
                "progress" => {
                    let (step, of) = (event["step"].as_u64().unwrap_or(0), event["total"].as_u64().unwrap_or(0));
                    out.event(json!({"event": "progress", "job": n, "step": step, "total": of, "elapsed": event["elapsed"]}));
                    if terminal {
                        eprint!("\r\x1b[2K  sampling {step}/{of}");
                        if step == of {
                            eprintln!();
                        }
                    } else if !out.json && (step % 10 == 0 || step == of) {
                        eprintln!("  sampling {step}/{of}");
                    }
                }
                _ => {}
            }
            true
        };

        match backend.request(&request, &mut on_event) {
            Ok(event) if event["event"] == "done" => {
                let result = LibResult {
                    id: library::new_id(),
                    ply: event["ply"].as_str().unwrap_or("").into(),
                    preprocessed: event["preprocessed"].as_str().unwrap_or("").into(),
                    camera_path: event["camera_path"].as_str().unwrap_or("").into(),
                    params: params.clone(),
                    seconds: event["seconds"].as_f64().unwrap_or(0.0),
                    num_gaussians: event["num_gaussians"].as_u64().unwrap_or(args.gaussians),
                    vram_peak_gb: event["vram_peak_gb"].as_f64(),
                    created: library::now_ms(),
                };
                if !args.no_library {
                    let preview = result.preprocessed.clone();
                    let stored = result.clone();
                    if let Err(e) = library::update(|lib| {
                        let i = library::ensure_image(lib, image);
                        lib[i].results.insert(0, stored);
                        lib[i].preview = Some(preview);
                    }) {
                        out.say(format!("  warning: could not add the result to the library: {e}"));
                    }
                }
                let summary = json!({
                    "event": "result", "job": n, "image": image, "seed": seed, "ply": result.ply,
                    "preprocessed": result.preprocessed, "camera_path": result.camera_path, "camera": event["camera"],
                    "seconds": result.seconds, "num_gaussians": result.num_gaussians, "vram_peak_gb": result.vram_peak_gb,
                    "params": params,
                });
                out.event(summary.clone());
                out.say(format!("  done in {}s: {}", result.seconds, result.ply));
                if !out.json {
                    println!("{}", result.ply);
                }
                results.push(summary);
            }
            Ok(event) => {
                let message = event["message"].as_str().unwrap_or("cancelled").to_string();
                failed += 1;
                out.event(json!({"event": "job_error", "job": n, "image": image, "seed": seed, "message": message}));
                out.say(format!("  failed: {message}"));
                if access_error(&message) {
                    exit = EXIT_ACCESS;
                    break; // every other job would fail the same way
                }
                exit = EXIT_ERROR;
            }
            Err(e) => {
                failed += 1;
                out.event(json!({"event": "job_error", "job": n, "image": image, "seed": seed, "message": e}));
                out.say(format!("  failed: {e}"));
                exit = EXIT_ERROR;
                break; // the worker is gone
            }
        }
    }
    backend.close();
    let skipped = total - results.len() - failed;
    out.event(json!({"event": "summary", "ok": results.len(), "failed": failed, "skipped": skipped, "results": results}));
    out.say(format!("{} ok, {failed} failed{}", results.len(), if skipped > 0 { format!(", {skipped} skipped") } else { String::new() }));
    exit
}

fn cmd_preview(args: PreviewArgs) -> i32 {
    let out = Out { json: args.pipeline.json };
    let image = absolute(&args.image);
    if !image.is_file() || !is_image(&image) {
        out.error(&format!("{}: not an image file", image.display()));
        return EXIT_USAGE;
    }
    let target = match args.out.clone() {
        Some(path) => absolute(&path),
        None => match settings::output_dir(&settings::load()) {
            Ok(dir) => dir.join(stem(&image)).join(format!("{}_preview_{}.png", stem(&image), args.mode.as_str())),
            Err(e) => {
                out.error(&e);
                return EXIT_ERROR;
            }
        },
    };
    let mut backend = match Backend::open(&args.pipeline, &out) {
        Ok(b) => b,
        Err(e) => {
            out.error(&e);
            return EXIT_ERROR;
        }
    };
    let request = merge(
        json!({"cmd": "preprocess", "id": format!("cli-{}", library::new_id()), "image": image, "out_path": target,
               "mode": args.mode.as_str(), "crop_margin": args.crop}),
        &pipeline_config(&args.pipeline),
    );
    let result = backend.request(&request, &mut |event| {
        if event["event"] == "status" {
            out.say(format!("  {}", event["message"].as_str().unwrap_or("")));
        }
        true
    });
    backend.close();
    match result {
        Ok(event) if event["event"] == "preprocessed" => {
            out.event(json!({"event": "preview", "image": image, "path": event["path"]}));
            if !out.json {
                println!("{}", event["path"].as_str().unwrap_or(""));
            }
            EXIT_OK
        }
        Ok(event) => {
            out.error(event["message"].as_str().unwrap_or("failed"));
            EXIT_ERROR
        }
        Err(e) => {
            out.error(&e);
            EXIT_ERROR
        }
    }
}

fn cmd_check_access(args: PipelineArgs) -> i32 {
    let out = Out { json: args.json };
    let mut backend = match Backend::open(&args, &out) {
        Ok(b) => b,
        Err(e) => {
            out.error(&e);
            return EXIT_ERROR;
        }
    };
    let request = merge(json!({"cmd": "check_access", "id": format!("cli-{}", library::new_id())}), &pipeline_config(&args));
    let result = backend.request(&request, &mut |_| true);
    backend.close();
    match result {
        Ok(event) if event["event"] == "access" => {
            let repos = event["repos"].as_array().cloned().unwrap_or_default();
            let all_ok = event["logged_in"].as_bool().unwrap_or(false) && repos.iter().all(|r| r["ok"].as_bool().unwrap_or(false));
            out.event(json!({"event": "access", "ok": all_ok, "logged_in": event["logged_in"], "repos": repos}));
            if !out.json {
                if !event["logged_in"].as_bool().unwrap_or(false) {
                    println!("No Hugging Face token: set one with `idol-studio-cli config set hf_token <token>` or `hf auth login`.");
                }
                for repo in &repos {
                    let ok = repo["ok"].as_bool().unwrap_or(false);
                    println!(
                        "{} {}{}",
                        if ok { "ok    " } else { "DENIED" },
                        repo["repo"].as_str().unwrap_or(""),
                        if ok { String::new() } else { format!("  (accept the terms at https://huggingface.co/{})", repo["repo"].as_str().unwrap_or("")) }
                    );
                }
            }
            if all_ok { EXIT_OK } else { EXIT_ACCESS }
        }
        Ok(event) => {
            out.error(event["message"].as_str().unwrap_or("failed"));
            EXIT_ERROR
        }
        Err(e) => {
            out.error(&e);
            EXIT_ERROR
        }
    }
}

fn cmd_info(args: InfoArgs) -> i32 {
    let out = Out { json: args.json };
    let settings = settings::load();
    let python = settings::python(&settings).map(|p| p.display().to_string()).unwrap_or_default();
    let server = RemoteWorker::connect().map(|(_, port)| port);
    let mut info = json!({
        "version": env!("CARGO_PKG_VERSION"),
        "settings_file": paths::config_dir().map(|d| d.join("settings.json")).unwrap_or_default(),
        "library_file": library::file().unwrap_or_default(),
        "python": python,
        "python_exists": Path::new(&python).exists(),
        "output_dir": settings::output_dir(&settings).unwrap_or_default(),
        "weights": settings.weights,
        "dinov3_dtype": settings.dinov3_dtype,
        "optimize": settings.optimize,
        "hf_token": if settings.hf_token.trim().is_empty() { Value::Null } else { json!(mask_token(&settings.hf_token)) },
        "server_port": server,
        "library_images": library::load().len(),
    });
    if !args.no_worker && Path::new(&python).exists() {
        let pipeline = PipelineArgs { weights: None, dtype: None, optimize: false, no_optimize: false, no_server: false, verbose: false, json: true };
        let quiet = Out { json: true };
        if let Ok(mut backend) = Backend::open(&pipeline, &quiet) {
            if let Ok(hello) = backend.request(&json!({"cmd": "hello", "id": "cli-hello"}), &mut |_| true) {
                for key in ["torch", "cuda", "gpu", "vram_gb", "idol", "triton", "import_error"] {
                    if !hello[key].is_null() {
                        info[key] = hello[key].clone();
                    }
                }
                info["python_version"] = hello["python"].clone();
            }
            backend.close();
        }
    }
    if out.json {
        out.event(merge(json!({"event": "info"}), &info));
    } else if let Some(map) = info.as_object() {
        for (key, value) in map {
            println!("{key:<16} {}", value.as_str().map(String::from).unwrap_or_else(|| value.to_string()));
        }
    }
    EXIT_OK
}

fn cmd_setup(args: JsonArg) -> i32 {
    let out = Out { json: args.json };
    let log = |line: String| {
        if out.json {
            out.event(json!({"event": "log", "line": line}));
        } else {
            eprintln!("{line}");
        }
    };
    match setup::setup(&log) {
        Ok(python) => {
            out.event(json!({"event": "setup", "ok": true, "python": python}));
            out.say("Environment is ready.");
            EXIT_OK
        }
        Err(e) => {
            out.error(&e);
            EXIT_ERROR
        }
    }
}

fn cmd_config(args: ConfigArgs) -> i32 {
    let out = Out { json: args.json };
    let mut settings = serde_json::to_value(settings::load()).unwrap_or_default();
    let show = |settings: &Value| -> Value {
        let mut shown = settings.clone();
        if !args.show_token {
            shown["hf_token"] = json!(mask_token(settings["hf_token"].as_str().unwrap_or("")));
        }
        shown
    };
    match args.action {
        None => {
            let shown = show(&settings);
            if out.json {
                out.event(merge(json!({"event": "config"}), &shown));
            } else {
                println!("{}", serde_json::to_string_pretty(&shown).unwrap_or_default());
            }
            EXIT_OK
        }
        Some(ConfigAction::Get { key }) => match settings.get(&key) {
            Some(_) => {
                let value = show(&settings)[&key].clone();
                if out.json {
                    out.event(json!({"event": "config", "key": key, "value": value}));
                } else {
                    println!("{}", value.as_str().map(String::from).unwrap_or_else(|| value.to_string()));
                }
                EXIT_OK
            }
            None => {
                out.error(&format!("unknown setting `{key}`"));
                EXIT_USAGE
            }
        },
        Some(ConfigAction::Set { key, value }) => {
            let Some(current) = settings.get(&key).cloned() else {
                out.error(&format!("unknown setting `{key}`"));
                return EXIT_USAGE;
            };
            let new_value = if current.is_boolean() {
                match value.to_lowercase().as_str() {
                    "true" | "on" | "yes" | "1" => json!(true),
                    "false" | "off" | "no" | "0" => json!(false),
                    _ => {
                        out.error(&format!("`{key}` is true or false"));
                        return EXIT_USAGE;
                    }
                }
            } else {
                json!(value)
            };
            if key == "dinov3_dtype" && !["bf16", "fp32"].contains(&value.as_str()) {
                out.error("dinov3_dtype is bf16 or fp32");
                return EXIT_USAGE;
            }
            settings[&key] = new_value;
            let parsed: settings::Settings = match serde_json::from_value(settings.clone()) {
                Ok(s) => s,
                Err(e) => {
                    out.error(&e.to_string());
                    return EXIT_USAGE;
                }
            };
            if let Err(e) = settings::save(&parsed) {
                out.error(&e);
                return EXIT_ERROR;
            }
            out.event(json!({"event": "config", "key": key, "value": show(&settings)[&key]}));
            out.say(format!("{key} saved."));
            EXIT_OK
        }
    }
}

fn cmd_library(args: JsonArg) -> i32 {
    let library = library::load();
    if args.json {
        println!("{}", json!({"event": "library", "images": library}));
        return EXIT_OK;
    }
    if library.is_empty() {
        println!("The library is empty.");
    }
    for image in &library {
        println!("{}  ({} result{})", image.path, image.results.len(), if image.results.len() == 1 { "" } else { "s" });
        for r in &image.results {
            println!("    seed {:<10} {:>9} gaussians  {:>7.1}s  {}", r.params.seed, r.num_gaussians, r.seconds, r.ply);
        }
    }
    EXIT_OK
}

/// Keep one worker loaded and serve requests from `generate` / `preview` / `check-access` over 127.0.0.1.
/// Connections are handled one at a time; others wait their turn.
fn cmd_serve(args: ServeArgs) -> i32 {
    let out = Out { json: args.pipeline.json };
    let listener = match TcpListener::bind(("127.0.0.1", args.port)) {
        Ok(l) => l,
        Err(e) => {
            out.error(&format!("could not listen on 127.0.0.1:{}: {e} (is a server already running?)", args.port));
            return EXIT_ERROR;
        }
    };
    let state = json!({"port": args.port, "pid": std::process::id(), "version": env!("CARGO_PKG_VERSION")});
    let Ok(state_file) = serve_file() else { return EXIT_ERROR };
    if let Err(e) = paths::write_atomic(&state_file, state.to_string().as_bytes()) {
        out.error(&e);
        return EXIT_ERROR;
    }
    let mut worker: Option<LocalWorker> = None;
    let pipeline = pipeline_config(&args.pipeline);
    let start = |out: &Out| -> Result<LocalWorker, String> {
        out.say("Starting the Python worker...");
        LocalWorker::start(args.pipeline.verbose)
    };

    if !args.lazy {
        match start(&out) {
            Ok(mut w) => {
                out.say("Loading the model...");
                let request = merge(json!({"cmd": "load", "id": "serve-load"}), &pipeline);
                match w.request(&request, &mut |e| {
                    if e["event"] == "status" {
                        out.say(format!("  {}", e["message"].as_str().unwrap_or("")));
                    }
                    true
                }) {
                    Ok(e) if e["event"] == "loaded" => out.say(format!("Model loaded in {}s.", e["seconds"])),
                    Ok(e) => out.say(format!("Could not load the model: {}", e["message"].as_str().unwrap_or(""))),
                    Err(e) => out.say(format!("Could not load the model: {e}")),
                }
                worker = Some(w);
            }
            Err(e) => {
                out.error(&e);
                let _ = std::fs::remove_file(&state_file);
                return EXIT_ERROR;
            }
        }
    }
    out.event(json!({"event": "serving", "port": args.port}));
    out.say(format!("Serving on 127.0.0.1:{}. `idol-studio-cli generate ...` uses this server. Ctrl+C to stop.", args.port));

    for stream in listener.incoming() {
        let Ok(stream) = stream else { continue };
        let _ = stream.set_nodelay(true);
        let Ok(read_half) = stream.try_clone() else { continue };
        let mut writer = stream;
        if writeln!(writer, "{}", json!({"event": "idol-studio-serve", "version": env!("CARGO_PKG_VERSION")})).is_err() {
            continue;
        }
        for line in BufReader::new(read_half).lines() {
            let Ok(line) = line else { break };
            let Ok(request) = serde_json::from_str::<Value>(&line) else { continue };
            if request["cmd"] == "cancel" {
                continue; // only meaningful while a request runs; handled below
            }
            if worker.as_mut().is_none_or(|w| matches!(w.child.try_wait(), Ok(Some(_)))) {
                match start(&out) {
                    Ok(w) => worker = Some(w),
                    Err(e) => {
                        let _ = writeln!(writer, "{}", json!({"event": "error", "id": request["id"], "message": e}));
                        continue;
                    }
                }
            }
            let w = worker.as_mut().expect("worker");
            out.say(format!("> {} {}", request["cmd"].as_str().unwrap_or("?"), request["image"].as_str().unwrap_or("")));
            let mut client = writer.try_clone().ok();
            let mut on_event = |event: &Value| -> bool {
                // A client that went away cancels its request.
                match client.as_mut() {
                    Some(c) => writeln!(c, "{event}").is_ok(),
                    None => false,
                }
            };
            let answer = match w.request(&request, &mut on_event) {
                Ok(event) => event,
                Err(e) => json!({"event": "error", "id": request["id"], "message": e}),
            };
            out.say(format!("< {}", answer["event"].as_str().unwrap_or("?")));
            if writeln!(writer, "{answer}").is_err() {
                break;
            }
        }
    }
    let _ = std::fs::remove_file(&state_file);
    EXIT_OK
}

// ---------------------------------------------------------------------------------------------------------------

/// Run the CLI with the process arguments. Returns the exit code.
pub fn run() -> i32 {
    let cli = match Cli::try_parse() {
        Ok(cli) => cli,
        Err(e) => {
            let _ = e.print();
            return if e.use_stderr() { EXIT_USAGE } else { EXIT_OK };
        }
    };
    match cli.command {
        Command::Generate(args) => cmd_generate(args),
        Command::Preview(args) => cmd_preview(args),
        Command::CheckAccess(args) => cmd_check_access(args),
        Command::Info(args) => cmd_info(args),
        Command::Setup(args) => cmd_setup(args),
        Command::Serve(args) => cmd_serve(args),
        Command::Config(args) => cmd_config(args),
        Command::Library(args) => cmd_library(args),
    }
}

/// For the GUI build (windows subsystem): when started from a terminal without redirected output, attach to
/// the terminal's console so output is visible. Redirected output (pipes, files) works without this.
pub fn attach_console() {
    #[cfg(windows)]
    unsafe {
        use windows_sys::Win32::System::Console::{AttachConsole, GetStdHandle, ATTACH_PARENT_PROCESS, STD_OUTPUT_HANDLE};
        let handle = GetStdHandle(STD_OUTPUT_HANDLE);
        if handle.is_null() || handle as isize == -1 {
            AttachConsole(ATTACH_PARENT_PROCESS);
        }
    }
}
