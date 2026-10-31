import { invoke, convertFileSrc } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { join, basename } from "@tauri-apps/api/path";
import { getCurrentWebview } from "@tauri-apps/api/webview";
import { open as openDialog, save as saveDialog } from "@tauri-apps/plugin-dialog";
import { openPath, openUrl, revealItemInDir } from "@tauri-apps/plugin-opener";
import { SplatViewer } from "./viewer";

// ---------------------------------------------------------------------------------------------------------------
// Types

type Mode = "synthetic" | "real";

interface GenParams {
  mode: Mode;
  crop_margin: number | null; // null: from mode
  seed: number;
  steps: number;
  cfg_strength: number;
  rescale_t: number;
  num_gaussians: number;
}

interface Result {
  id: string;
  ply: string;
  preprocessed: string;
  camera_path: string;
  params: GenParams;
  seconds: number;
  num_gaussians: number;
  vram_peak_gb?: number;
  created: number;
  exports?: { format: string; path: string }[];
}

interface LibraryImage {
  id: string;
  path: string;
  name: string;
  results: Result[];
  preview?: string; // latest preprocessed preview
}

type JobStatus = "queued" | "running" | "done" | "error" | "cancelled";

interface Job {
  id: string;
  kind: "generate" | "preprocess" | "export";
  /** For export jobs. */
  export?: { ply: string; format: string; out_path: string | null; options: ExportOptions };
  imageId: string;
  imageName: string;
  params: GenParams;
  status: JobStatus;
  stage: string;
  step: number;
  total: number;
  message: string;
  started?: number;
}

interface ExportOptions {
  method: "poisson" | "density";
  resolution: number;
  threshold: number;
  smooth: number;
  faces: number;
  height: number | null;
}

const MESH_FORMATS = ["glb", "obj", "stl", "mesh-ply"];
const EXPORT_SUFFIX: Record<string, string> = {
  ply: "_splat.ply", spz: ".spz", splat: ".splat", points: "_points.ply", glb: ".glb", obj: ".obj", stl: ".stl", "mesh-ply": "_mesh.ply",
};

interface Settings {
  python_path: string;
  output_dir: string;
  hf_token: string;
  weights: string;
  dinov3_dtype: string;
  optimize: boolean;
  check_updates: boolean;
}

interface AppInfo {
  version: string;
  repo: string;
  env_dir: string;
  managed_python: string;
  python: string;
  python_exists: boolean;
  output_dir: string;
  worker_script: string;
  cli_path: string;
  cli_installed: boolean;
}

interface UpdateInfo {
  current: string;
  latest: string | null;
  available: boolean;
  notes: string;
  page_url: string;
  asset_url: string | null;
  asset_name: string | null;
  checked_at: number;
}

type WorkerEvent = { event: string; id: string | null; [key: string]: any };

// ---------------------------------------------------------------------------------------------------------------
// State

const DEFAULTS: GenParams = {
  mode: "synthetic",
  crop_margin: null,
  seed: 42,
  steps: 50,
  cfg_strength: 3.0,
  rescale_t: 3.0,
  num_gaussians: 262144,
};
const CROP_FOR_MODE: Record<Mode, number> = { synthetic: 1.2, real: 1.7 };
const LIBRARY_KEY = "idol-studio.library.v1";
const OPTIONS_KEY = "idol-studio.options.v1";

// The library lives in library.json (shared with the CLI); this is the page's copy.
let library: LibraryImage[] = [];
let options: GenParams & { variants: number; randomSeed: boolean } = {
  ...DEFAULTS,
  variants: 1,
  randomSeed: false,
  ...loadJson(OPTIONS_KEY, {}),
};
let settings: Settings;
let info: AppInfo;
let selectedImageId: string | null = null;
let selectedResultId: string | null = null;
let jobs: Job[] = [];
let workerState: "stopped" | "starting" | "idle" | "loading" | "busy" = "stopped";
let modelLoaded = false;
let pendingUpdate: UpdateInfo | null = null;
// Requests waiting for an answer: return true once the event settled the request.
const pendingRequests = new Map<string, (event: WorkerEvent) => boolean>();

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
const viewer = new SplatViewer($("viewer-host"));
// Dev builds only: lets test scripts drive the viewer over the WebView2 debugging port.
if (import.meta.env.DEV) (window as any).__idolViewer = viewer;

function loadJson<T>(key: string, fallback: T): T {
  try {
    const text = localStorage.getItem(key);
    return text ? (JSON.parse(text) as T) : fallback;
  } catch {
    return fallback;
  }
}
/** Change the library safely: reload it from disk (the CLI may have added results), change it, save it. */
async function mutateLibrary(change: (library: LibraryImage[]) => void) {
  library = await invoke<LibraryImage[]>("library_load");
  change(library);
  await invoke("library_save", { library });
}
const saveOptions = () => localStorage.setItem(OPTIONS_KEY, JSON.stringify(options));
const uid = () => Math.random().toString(36).slice(2, 10) + Date.now().toString(36);
const fmt = (n: number) => n.toLocaleString("en-US");
const escapeHtml = (s: string) =>
  s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]!);
const fileUrl = (path: string, bust?: number) => convertFileSrc(path) + (bust ? `?t=${bust}` : "");

// ---------------------------------------------------------------------------------------------------------------
// Log

const logEl = $("log");
const MAX_LOG_LINES = 3000;
function log(line: string, replaceLast = false) {
  if (replaceLast && logEl.lastChild) {
    logEl.lastChild.textContent = line + "\n";
  } else {
    const span = document.createElement("span");
    span.textContent = line + "\n";
    logEl.appendChild(span);
    while (logEl.childNodes.length > MAX_LOG_LINES) logEl.firstChild?.remove();
  }
  logEl.scrollTop = logEl.scrollHeight;
}

// ---------------------------------------------------------------------------------------------------------------
// Worker

function setWorkerState(state: typeof workerState, detail = "") {
  workerState = state;
  const pill = $("worker-pill");
  const labels = { stopped: "stopped", starting: "starting...", idle: modelLoaded ? "model loaded" : "ready", loading: "loading model...", busy: "generating" };
  pill.textContent = `Worker: ${labels[state]}${detail ? " · " + detail : ""}`;
  pill.className = "pill " + state;
}

async function send(request: Record<string, unknown>) {
  if (workerState === "stopped") setWorkerState("starting");
  try {
    await invoke("worker_send", { request });
  } catch (error) {
    setWorkerState("stopped");
    throw error;
  }
}

/** Send a request and wait for the event that answers it. */
function ask(cmd: string, extra: Record<string, unknown> = {}, answer: string): Promise<WorkerEvent> {
  const id = "q-" + uid();
  return new Promise((resolve, reject) => {
    pendingRequests.set(id, (event) => {
      if (event.event === "error") reject(new Error(event.message));
      else if (event.event === answer) resolve(event);
      else return false;
      return true;
    });
    send({ cmd, id, ...extra }).catch(reject);
  });
}

function pipelineConfig() {
  return { weights: settings.weights, dinov3_dtype: settings.dinov3_dtype, optimize: settings.optimize };
}

let gpuLabel = "";
function onWorkerEvent(event: WorkerEvent) {
  const pending = event.id ? pendingRequests.get(event.id) : undefined;
  if (pending && pending(event)) pendingRequests.delete(event.id!);
  const job = event.id ? jobs.find((j) => j.id === event.id) : undefined;

  switch (event.event) {
    case "ready":
      setWorkerState("idle");
      log(`[worker] started (pid ${event.pid})`);
      break;
    case "hello":
      if (event.gpu) gpuLabel = `${event.gpu} ${event.vram_gb} GB`;
      setWorkerState(workerState === "starting" ? "idle" : workerState, gpuLabel);
      break;
    case "status":
      log(`[worker] ${event.message}`);
      if (event.stage === "load") setWorkerState("loading");
      if (job) {
        job.stage = event.stage;
        job.message = event.message;
        renderQueue();
      }
      break;
    case "loaded":
      modelLoaded = true;
      log(`[worker] model loaded in ${event.seconds}s (${event.vram_allocated_gb ?? "?"} GB VRAM)`);
      setWorkerState(jobs.some((j) => j.status === "running") ? "busy" : "idle", gpuLabel);
      break;
    case "unloaded":
      modelLoaded = false;
      setWorkerState("idle", gpuLabel);
      break;
    case "progress":
      if (job) {
        if (workerState !== "busy") setWorkerState("busy", gpuLabel);
        job.stage = "sample";
        job.step = event.step;
        job.total = event.total;
        job.message = `Sampling ${event.step}/${event.total}`;
        renderQueueItem(job);
      }
      break;
    case "done":
      if (job) finishGenerate(job, event);
      else adoptResult(event);
      break;
    case "exported":
      if (job) finishExport(job, event);
      break;
    case "preprocessed":
      if (job) finishPreprocess(job, event);
      break;
    case "cancelled":
      if (job) endJob(job, "cancelled", "Cancelled");
      break;
    case "error":
      log(`[error] ${event.message}`);
      if (job) endJob(job, "error", event.message);
      else if (!pending) toast(event.message, true);
      break;
  }
}

function endJob(job: Job, status: JobStatus, message: string) {
  job.status = status;
  job.message = message;
  renderQueue();
  setWorkerState("idle", gpuLabel);
  pump();
}

async function finishGenerate(job: Job, event: WorkerEvent) {
  const result: Result = {
    id: uid(),
    ply: event.ply,
    preprocessed: event.preprocessed,
    camera_path: event.camera_path,
    params: job.params,
    seconds: event.seconds,
    num_gaussians: event.num_gaussians,
    vram_peak_gb: event.vram_peak_gb,
    created: Date.now(),
  };
  await mutateLibrary((lib) => {
    const image = lib.find((i) => i.id === job.imageId);
    if (image) {
      image.results.unshift(result);
      image.preview = event.preprocessed;
    }
  }).catch((e) => log(`[library] ${e}`));
  const image = library.find((i) => i.id === job.imageId);
  log(`[done] ${job.imageName}: ${fmt(event.num_gaussians)} Gaussians in ${event.seconds}s, peak ${event.vram_peak_gb ?? "?"} GB VRAM`);
  endJob(job, "done", `${event.seconds}s · ${fmt(event.num_gaussians)} Gaussians`);
  if (image && image.id === selectedImageId) {
    selectedResultId = result.id;
    renderLibrary();
    renderResults();
    showResult(result);
  } else {
    renderLibrary();
  }
}

async function finishExport(job: Job, event: WorkerEvent) {
  await invoke("library_add_export", { ply: event.ply, format: event.format, path: event.path }).catch((e) => log(`[library] ${e}`));
  library = await invoke<LibraryImage[]>("library_load");
  const mb = (event.bytes / 2 ** 20).toFixed(1);
  const mesh = event.faces ? ` · ${fmt(event.faces)} triangles` : "";
  log(`[export] ${event.format}: ${event.path} (${mb} MB${mesh}, ${event.seconds}s)`);
  endJob(job, "done", `${mb} MB${mesh} · ${event.path}`);
  toast(`Exported .${event.format}: ${event.path}`);
  const result = selectedResult();
  if (result && result.ply.toLowerCase() === String(event.ply).toLowerCase()) {
    renderResults();
    fillShowMenu(result);
    if (event.format === "glb") {
      ($("vt-show") as HTMLSelectElement).value = event.path;
      showResult(result, event.path);
    }
  }
}

/** A result for a job this page doesn't know (e.g. the page was reloaded mid-run): file it under its image. */
function adoptResult(event: WorkerEvent) {
  const image = library.find((i) => i.path === event.image);
  if (!image || !event.params) return;
  const job: Job = {
    id: event.id ?? uid(), kind: "generate", imageId: image.id, imageName: image.name, params: event.params,
    status: "running", stage: "", step: 0, total: 0, message: "",
  };
  jobs.push(job);
  finishGenerate(job, event);
}

async function finishPreprocess(job: Job, event: WorkerEvent) {
  await mutateLibrary((lib) => {
    const image = lib.find((i) => i.id === job.imageId);
    if (image) image.preview = event.path;
  }).catch((e) => log(`[library] ${e}`));
  const image = library.find((i) => i.id === job.imageId);
  endJob(job, "done", "Preview ready");
  if (image && image.id === selectedImageId) {
    showPreprocessed(event.path);
    switchTab("pre");
  }
}

// ---------------------------------------------------------------------------------------------------------------
// Queue

async function outBase(image: LibraryImage, params: GenParams) {
  const stem = image.name.replace(/\.[^.]+$/, "").replace(/[^\w.-]+/g, "_");
  const stamp = new Date().toISOString().replace(/[-:]/g, "").replace(/\..+/, "").replace("T", "-");
  const k = Math.round(params.num_gaussians / 1024);
  return join(info.output_dir, stem, `${stem}_${params.mode}_s${params.seed}_${k}k_${stamp}`);
}

async function pump() {
  if (jobs.some((j) => j.status === "running")) return;
  const job = jobs.find((j) => j.status === "queued");
  if (!job) {
    if (workerState === "busy") setWorkerState("idle", gpuLabel);
    return;
  }
  const image = library.find((i) => i.id === job.imageId);
  if (!image) {
    endJob(job, "error", "Image was removed");
    return;
  }
  job.status = "running";
  job.started = Date.now();
  const needsLoad = job.kind === "generate" && !modelLoaded;
  job.message = needsLoad ? "Loading model (the first run downloads several GB of weights)..." : "Starting...";
  renderQueue();
  setWorkerState(needsLoad ? "loading" : "busy", gpuLabel);
  try {
    const common = { id: job.id, image: image.path, mode: job.params.mode, crop_margin: job.params.crop_margin, ...pipelineConfig() };
    if (job.kind === "export" && job.export) {
      const e = job.export;
      await send({ cmd: "export", id: job.id, ply: e.ply, format: e.format, out_path: e.out_path, ...e.options });
    } else if (job.kind === "preprocess") {
      const out = await join(info.output_dir, ".previews", `${image.id}_${job.params.mode}_${job.params.crop_margin ?? "auto"}.png`);
      await send({ cmd: "preprocess", ...common, out_path: out });
    } else {
      await send({ cmd: "generate", ...common, ...job.params, out_base: await outBase(image, job.params) });
    }
  } catch (error) {
    endJob(job, "error", String(error));
  }
}

function currentParams(seedOffset = 0): GenParams {
  return {
    mode: options.mode,
    crop_margin: options.crop_margin,
    seed: options.seed + seedOffset,
    steps: options.steps,
    cfg_strength: options.cfg_strength,
    rescale_t: options.rescale_t,
    num_gaussians: options.num_gaussians,
  };
}

function enqueue(images: LibraryImage[], kind: Job["kind"]) {
  if (!images.length) {
    toast("Add an image first.");
    return;
  }
  for (const image of images) {
    const count = kind === "preprocess" ? 1 : options.variants;
    for (let v = 0; v < count; v++) {
      if (kind === "generate" && options.randomSeed) {
        options.seed = Math.floor(Math.random() * 2 ** 31);
      }
      const params = currentParams(kind === "generate" && !options.randomSeed ? v : 0);
      jobs.push({ id: "j-" + uid(), kind, imageId: image.id, imageName: image.name, params, status: "queued", stage: "", step: 0, total: 0, message: "Queued" });
    }
  }
  if (options.randomSeed) syncOptionsToForm();
  renderQueue();
  switchDock("queue");
  pump();
}

async function cancelJob(job: Job) {
  if (job.status === "queued") {
    job.status = "cancelled";
    job.message = "Cancelled";
    renderQueue();
  } else if (job.status === "running") {
    job.message = "Cancelling...";
    renderQueueItem(job);
    await send({ cmd: "cancel" }).catch(() => {});
  }
}

const queueEl = $("queue");
function renderQueue() {
  const active = jobs.filter((j) => j.status === "queued" || j.status === "running").length;
  $("queue-count").textContent = String(active);
  queueEl.innerHTML = "";
  if (!jobs.length) {
    queueEl.innerHTML = `<li class="empty">Nothing queued. Pick an image and press Generate.</li>`;
    return;
  }
  for (const job of [...jobs].reverse()) {
    const li = document.createElement("li");
    li.id = "job-" + job.id;
    queueEl.appendChild(li);
    renderQueueItem(job);
  }
}

function renderQueueItem(job: Job) {
  const li = document.getElementById("job-" + job.id);
  if (!li) return;
  const pct = job.status === "done" ? 100 : job.total ? Math.round((job.step / job.total) * 100) : 0;
  const p = job.params;
  const what =
    job.kind === "export"
      ? `Export .${job.export?.format} · seed ${p.seed}`
      : job.kind === "preprocess"
        ? "Preview crop"
        : `seed ${p.seed} · ${fmt(p.num_gaussians)} · ${p.steps} steps · ${p.mode}`;
  li.className = "job " + job.status;
  li.innerHTML = `
    <div class="job-main">
      <strong>${escapeHtml(job.imageName)}</strong> <span class="muted">${what}</span>
      <div class="bar"><div style="width:${pct}%"></div></div>
      <span class="job-msg">${escapeHtml(job.message)}</span>
    </div>
    <span class="job-status">${job.status}</span>
    ${job.status === "queued" || job.status === "running" ? `<button class="small ghost" data-cancel="${job.id}">Cancel</button>` : ""}`;
  li.querySelector<HTMLButtonElement>("[data-cancel]")?.addEventListener("click", () => cancelJob(job));
}

// ---------------------------------------------------------------------------------------------------------------
// Library

const listEl = $("image-list");
function renderLibrary() {
  listEl.innerHTML = "";
  if (!library.length) {
    listEl.innerHTML = `<li class="empty">No images yet.</li>`;
  }
  for (const image of library) {
    const li = document.createElement("li");
    li.className = "image-item" + (image.id === selectedImageId ? " selected" : "");
    li.innerHTML = `
      <img src="${fileUrl(image.path)}" alt="" loading="lazy" />
      <div class="image-meta">
        <span class="name" title="${escapeHtml(image.path)}">${escapeHtml(image.name)}</span>
        <span class="muted">${image.results.length ? `${image.results.length} result${image.results.length > 1 ? "s" : ""}` : "not generated"}</span>
      </div>
      <button class="icon-btn" title="Remove from list" data-remove>&times;</button>`;
    li.addEventListener("click", () => selectImage(image.id));
    li.querySelector("[data-remove]")!.addEventListener("click", async (e) => {
      e.stopPropagation();
      await mutateLibrary((lib) => lib.splice(0, lib.length, ...lib.filter((i) => i.id !== image.id)));
      if (selectedImageId === image.id) selectedImageId = library[0]?.id ?? null;
      selectImage(selectedImageId);
    });
    listEl.appendChild(li);
  }
}

async function addImages(paths: string[]) {
  const fresh: LibraryImage[] = [];
  for (const path of paths) {
    let files = [path];
    if (await invoke<boolean>("is_dir", { path })) files = await invoke<string[]>("list_images", { dir: path });
    for (const file of files) {
      if (!/\.(png|jpe?g|webp|bmp)$/i.test(file)) continue;
      fresh.push({ id: uid(), path: file, name: await basename(file), results: [] });
    }
  }
  let added = 0;
  let firstId = "";
  await mutateLibrary((lib) => {
    for (const image of fresh) {
      if (lib.some((i) => i.path.toLowerCase() === image.path.toLowerCase())) continue;
      lib.push(image);
      firstId ||= image.id;
      added++;
    }
  });
  if (added) {
    selectImage(firstId);
    toast(`Added ${added} image${added > 1 ? "s" : ""}.`);
  } else {
    renderLibrary();
  }
}

const selectedImage = () => library.find((i) => i.id === selectedImageId);

function selectImage(id: string | null) {
  selectedImageId = id;
  const image = selectedImage();
  selectedResultId = image?.results[0]?.id ?? null;
  renderLibrary();
  renderResults();
  const input = $<HTMLImageElement>("img-input");
  if (image) input.src = fileUrl(image.path);
  else input.removeAttribute("src");
  const result = image?.results[0];
  if (result) showResult(result);
  else {
    viewer.clear();
    $("viewer-empty").classList.remove("hidden");
    $("viewer-stats").textContent = "";
    showPreprocessed(image?.preview);
  }
}

// ---------------------------------------------------------------------------------------------------------------
// Results

function renderResults() {
  const image = selectedImage();
  const strip = $("result-strip");
  $("results-for").textContent = image ? `· ${image.name}` : "";
  strip.innerHTML = "";
  if (!image || !image.results.length) {
    strip.innerHTML = `<span class="muted">No results yet.</span>`;
  }
  for (const result of image?.results ?? []) {
    const p = result.params;
    const card = document.createElement("button");
    card.className = "result-card" + (result.id === selectedResultId ? " selected" : "");
    card.innerHTML = `
      <img src="${fileUrl(result.preprocessed)}" alt="" loading="lazy" />
      <span><b>seed ${p.seed}</b> · ${p.mode}</span>
      <span class="muted">${fmt(result.num_gaussians)} · ${p.steps} st · cfg ${p.cfg_strength}</span>
      <span class="muted">${result.seconds}s${result.vram_peak_gb ? ` · ${result.vram_peak_gb} GB` : ""}</span>
      ${result.exports?.length ? `<span>${[...new Set(result.exports.map((e) => e.format))].map((f) => `<span class="export-chip">${escapeHtml(f)}</span>`).join("")}</span>` : ""}`;
    card.addEventListener("click", () => {
      selectedResultId = result.id;
      renderResults();
      showResult(result);
    });
    strip.appendChild(card);
  }
  const has = !!selectedResult();
  for (const id of ["btn-open-ply", "btn-reveal", "btn-export", "btn-use-params", "btn-delete-result"]) $<HTMLButtonElement>(id).disabled = !has;
}

const selectedResult = () => selectedImage()?.results.find((r) => r.id === selectedResultId);

function showPreprocessed(path?: string) {
  const img = $<HTMLImageElement>("img-pre");
  if (path) {
    img.src = fileUrl(path, Date.now());
    img.classList.remove("hidden");
    $("pre-empty").classList.add("hidden");
  } else {
    img.removeAttribute("src");
    img.classList.add("hidden");
    $("pre-empty").classList.remove("hidden");
  }
}

/** The viewer's "show" menu: the splat, plus any exported .glb meshes of this result. */
function fillShowMenu(result: Result | undefined) {
  const select = $<HTMLSelectElement>("vt-show");
  const current = select.value;
  const meshes = (result?.exports ?? []).filter((e) => e.format === "glb");
  select.innerHTML = `<option value="">Splat</option>` + meshes.map((m) => `<option value="${escapeHtml(m.path)}">Mesh (.glb)${meshes.length > 1 ? " " + escapeHtml(m.path.split(/[\\/]/).pop()!) : ""}</option>`).join("");
  select.value = meshes.some((m) => m.path === current) ? current : "";
  select.classList.toggle("hidden", meshes.length === 0);
}

async function showResult(result: Result, meshPath = "") {
  showPreprocessed(result.preprocessed);
  if (!meshPath) fillShowMenu(result);
  $("viewer-empty").classList.add("hidden");
  $("viewer-loading").classList.remove("hidden");
  $("viewer-stats").textContent = "";
  try {
    let cam: { yaw: number; pitch: number; fov: number } | null = null;
    try {
      cam = JSON.parse(await invoke<string>("read_text_file", { path: result.camera_path }));
    } catch {}
    if (meshPath) {
      const meshBytes = await invoke<ArrayBuffer>("read_file_bytes", { path: meshPath });
      const triangles = await viewer.loadMesh(meshBytes, cam);
      if (triangles === null) return;
      $("viewer-stats").textContent = `Mesh preview · ${fmt(triangles)} triangles · ${(meshBytes.byteLength / 2 ** 20).toFixed(1)} MB`;
      return;
    }
    const bytes = await invoke<ArrayBuffer>("read_file_bytes", { path: result.ply });
    const count = await viewer.load(bytes, cam);
    if (count === null) return;
    const deg = (r: number) => ((r * 180) / Math.PI).toFixed(0);
    const camera = cam ? ` · input camera yaw ${deg(cam.yaw)}°, pitch ${deg(cam.pitch)}°` : "";
    $("viewer-stats").textContent = `${fmt(count || result.num_gaussians)} splats · ${(bytes.byteLength / 2 ** 20).toFixed(1)} MB${camera}`;
  } catch (error) {
    viewer.clear();
    $("viewer-empty").classList.remove("hidden");
    $("viewer-empty").textContent = `Could not load the splat: ${error}`;
  } finally {
    $("viewer-loading").classList.add("hidden");
  }
}

// ---------------------------------------------------------------------------------------------------------------
// Generation form

function syncOptionsToForm() {
  document.querySelectorAll<HTMLButtonElement>("#opt-mode button").forEach((b) => b.classList.toggle("active", b.dataset.value === options.mode));
  const auto = options.crop_margin === null;
  $<HTMLInputElement>("opt-crop-auto").checked = auto;
  const crop = $<HTMLInputElement>("opt-crop");
  crop.disabled = auto;
  crop.value = String(options.crop_margin ?? CROP_FOR_MODE[options.mode]);
  $("out-crop").textContent = Number(crop.value).toFixed(2);
  $<HTMLInputElement>("opt-seed").value = String(options.seed);
  $<HTMLInputElement>("opt-seed-random").checked = options.randomSeed;
  $<HTMLInputElement>("opt-variants").value = String(options.variants);
  $("out-variants").textContent = String(options.variants);
  $<HTMLSelectElement>("opt-gaussians").value = String(options.num_gaussians);
  $<HTMLInputElement>("opt-steps").value = String(options.steps);
  $("out-steps").textContent = String(options.steps);
  $<HTMLInputElement>("opt-cfg").value = String(options.cfg_strength);
  $("out-cfg").textContent = options.cfg_strength.toFixed(1);
  $<HTMLInputElement>("opt-shift").value = String(options.rescale_t);
  $("out-shift").textContent = options.rescale_t.toFixed(1);
  if (settings) {
    $<HTMLSelectElement>("opt-dtype").value = settings.dinov3_dtype;
    $<HTMLInputElement>("opt-optimize").checked = settings.optimize;
    $<HTMLInputElement>("opt-weights").value = settings.weights;
  }
}

function bindForm() {
  document.querySelectorAll<HTMLButtonElement>("#opt-mode button").forEach((button) =>
    button.addEventListener("click", () => {
      options.mode = button.dataset.value as Mode;
      saveOptions();
      syncOptionsToForm();
    }),
  );
  $<HTMLInputElement>("opt-crop-auto").addEventListener("change", (e) => {
    options.crop_margin = (e.target as HTMLInputElement).checked ? null : CROP_FOR_MODE[options.mode];
    saveOptions();
    syncOptionsToForm();
  });
  const numeric: [string, keyof typeof options, (v: number) => number][] = [
    ["opt-crop", "crop_margin", (v) => v],
    ["opt-seed", "seed", (v) => Math.max(0, Math.floor(v || 0))],
    ["opt-variants", "variants", (v) => v],
    ["opt-gaussians", "num_gaussians", (v) => v],
    ["opt-steps", "steps", (v) => v],
    ["opt-cfg", "cfg_strength", (v) => v],
    ["opt-shift", "rescale_t", (v) => v],
  ];
  for (const [id, key, clean] of numeric) {
    $(id).addEventListener("input", (e) => {
      (options as any)[key] = clean(Number((e.target as HTMLInputElement).value));
      saveOptions();
      syncOptionsToForm();
    });
  }
  $<HTMLInputElement>("opt-seed-random").addEventListener("change", (e) => {
    options.randomSeed = (e.target as HTMLInputElement).checked;
    saveOptions();
  });
  $("btn-dice").addEventListener("click", () => {
    options.seed = Math.floor(Math.random() * 2 ** 31);
    saveOptions();
    syncOptionsToForm();
  });
  $("btn-defaults").addEventListener("click", () => {
    options = { ...DEFAULTS, variants: 1, randomSeed: false };
    saveOptions();
    syncOptionsToForm();
  });

  // Model options live in the app settings (they decide how the pipeline is loaded).
  const saveModel = async () => {
    settings.dinov3_dtype = $<HTMLSelectElement>("opt-dtype").value;
    settings.optimize = $<HTMLInputElement>("opt-optimize").checked;
    settings.weights = $<HTMLInputElement>("opt-weights").value.trim() || "speridlabs/idol";
    await invoke("save_settings", { newSettings: settings });
  };
  $("opt-dtype").addEventListener("change", saveModel);
  $("opt-optimize").addEventListener("change", saveModel);
  $("opt-weights").addEventListener("change", saveModel);
  $("btn-weights-browse").addEventListener("click", async () => {
    const dir = await openDialog({ directory: true, title: "Folder with Idol weights (config.json + .safetensors)" });
    if (typeof dir === "string") {
      $<HTMLInputElement>("opt-weights").value = dir;
      await saveModel();
    }
  });

  $("btn-generate").addEventListener("click", () => enqueue(selectedImage() ? [selectedImage()!] : [], "generate"));
  $("btn-generate-all").addEventListener("click", () => enqueue(library, "generate"));
  $("btn-preview").addEventListener("click", () => enqueue(selectedImage() ? [selectedImage()!] : [], "preprocess"));
}

// ---------------------------------------------------------------------------------------------------------------
// Tabs, dock, viewer tools

function switchTab(tab: string) {
  document.querySelectorAll<HTMLButtonElement>(".tabs .tab").forEach((b) => b.classList.toggle("active", b.dataset.tab === tab));
  $("view-3d").classList.toggle("hidden", tab !== "3d");
  $("view-input").classList.toggle("hidden", tab !== "input");
  $("view-pre").classList.toggle("hidden", tab !== "pre");
  $("viewer-tools").classList.toggle("invisible", tab !== "3d");
}

function switchDock(which: string) {
  document.querySelectorAll<HTMLButtonElement>(".dock-tabs .tab").forEach((b) => b.classList.toggle("active", b.dataset.dock === which));
  $("queue").classList.toggle("hidden", which !== "queue");
  $("log").classList.toggle("hidden", which !== "log");
  if (which === "log") logEl.scrollTop = logEl.scrollHeight;
}

function bindChrome() {
  document.querySelectorAll<HTMLButtonElement>(".tabs .tab").forEach((b) => b.addEventListener("click", () => switchTab(b.dataset.tab!)));
  document.querySelectorAll<HTMLButtonElement>(".dock-tabs .tab").forEach((b) => b.addEventListener("click", () => switchDock(b.dataset.dock!)));

  const rotate = $<HTMLInputElement>("vt-rotate");
  const flip = $<HTMLInputElement>("vt-flip");
  const bg = $<HTMLSelectElement>("vt-bg");
  const view = loadJson("idol-studio.view.v1", { rotate: true, flip: false, bg: "#15161c" });
  rotate.checked = view.rotate;
  flip.checked = view.flip;
  bg.value = view.bg;
  const applyView = () => {
    viewer.setAutoRotate(rotate.checked);
    viewer.setFlip(flip.checked);
    viewer.setBackground(bg.value);
    localStorage.setItem("idol-studio.view.v1", JSON.stringify({ rotate: rotate.checked, flip: flip.checked, bg: bg.value }));
  };
  [rotate, flip, bg].forEach((el) => el.addEventListener("change", applyView));
  applyView();
  $("vt-reset").addEventListener("click", () => viewer.resetView());

  $("btn-add-images").addEventListener("click", async () => {
    const picked = await openDialog({ multiple: true, filters: [{ name: "Images", extensions: ["png", "jpg", "jpeg", "webp", "bmp"] }] });
    if (picked) addImages(Array.isArray(picked) ? picked : [picked]);
  });
  $("btn-add-folder").addEventListener("click", async () => {
    const dir = await openDialog({ directory: true });
    if (typeof dir === "string") addImages([dir]);
  });

  // Drag and drop from Explorer (Tauri gives real file paths).
  const drop = $("dropzone");
  getCurrentWebview().onDragDropEvent((event) => {
    const type = event.payload.type;
    document.body.classList.toggle("dragging", type === "enter" || type === "over");
    if (type === "drop") addImages(event.payload.paths);
  });
  drop.addEventListener("click", () => $("btn-add-images").click());

  $("btn-open-ply").addEventListener("click", () => selectedResult() && openPath(selectedResult()!.ply).catch((e) => toast(String(e), true)));
  $("btn-reveal").addEventListener("click", () => selectedResult() && revealItemInDir(selectedResult()!.ply));
  bindExport();
  $("vt-show").addEventListener("change", () => {
    const result = selectedResult();
    if (result) showResult(result, $<HTMLSelectElement>("vt-show").value);
  });
  $("btn-use-params").addEventListener("click", () => {
    const result = selectedResult();
    if (!result) return;
    options = { ...options, ...result.params };
    saveOptions();
    syncOptionsToForm();
    toast("Settings copied to the panel.");
  });
  $("btn-delete-result").addEventListener("click", async () => {
    const image = selectedImage();
    const result = selectedResult();
    if (!image || !result) return;
    if (!confirm("Delete this result's files from disk?")) return;
    await invoke("delete_outputs", { paths: [result.ply, result.preprocessed, result.camera_path] }).catch((e) => toast(String(e), true));
    await mutateLibrary((lib) => {
      const stored = lib.find((i) => i.id === image.id);
      if (stored) stored.results = stored.results.filter((r) => r.id !== result.id);
    });
    selectImage(image.id);
  });

  $("btn-clear-done").addEventListener("click", () => {
    jobs = jobs.filter((j) => j.status === "queued" || j.status === "running");
    renderQueue();
  });
  $("btn-cancel-all").addEventListener("click", () => jobs.forEach((j) => cancelJob(j)));
  $("btn-unload").addEventListener("click", () => workerState !== "stopped" && send({ cmd: "unload" }));
  $("btn-stop-worker").addEventListener("click", stopWorker);
}

// ---------------------------------------------------------------------------------------------------------------
// Export

const EXPORT_OPTIONS_KEY = "idol-studio.export.v1";

function bindExport() {
  const dialog = $<HTMLDialogElement>("export-dialog");
  const saved = loadJson<{ format: string; options: ExportOptions; saveAs: boolean }>(EXPORT_OPTIONS_KEY, {
    format: "spz",
    options: { method: "poisson", resolution: 256, threshold: 0.5, smooth: 3, faces: 200000, height: null },
    saveAs: false,
  });
  if (!saved.options.method) {
    // Older saved settings predate Poisson; the old 10-pass default would over-smooth it.
    saved.options.method = "poisson";
    saved.options.smooth = 3;
  }
  saved.options.smooth ??= 3;
  const formatInputs = () => [...dialog.querySelectorAll<HTMLInputElement>('input[name="export-format"]')];
  const format = () => formatInputs().find((i) => i.checked)?.value ?? "spz";
  const method = () => ($<HTMLSelectElement>("ex-method").value as ExportOptions["method"]) ?? "poisson";
  const syncMesh = () => {
    const mesh = MESH_FORMATS.includes(format());
    $("mesh-options").classList.toggle("hidden", !mesh);
    $("field-threshold").classList.toggle("hidden", !mesh || method() !== "density");
  };
  const syncOutputs = () => {
    $("out-threshold").textContent = Number($<HTMLInputElement>("ex-threshold").value).toFixed(2);
    $("out-smooth").textContent = $<HTMLInputElement>("ex-smooth").value;
  };
  formatInputs().forEach((i) => i.addEventListener("change", syncMesh));
  $("ex-method").addEventListener("change", syncMesh);
  ["ex-threshold", "ex-smooth"].forEach((id) => $(id).addEventListener("input", syncOutputs));

  $("btn-export").addEventListener("click", () => {
    const result = selectedResult();
    if (!result) return;
    formatInputs().forEach((i) => (i.checked = i.value === saved.format));
    $<HTMLSelectElement>("ex-method").value = saved.options.method;
    $<HTMLSelectElement>("ex-resolution").value = String(saved.options.resolution);
    $<HTMLSelectElement>("ex-faces").value = String(saved.options.faces);
    $<HTMLInputElement>("ex-threshold").value = String(saved.options.threshold);
    $<HTMLInputElement>("ex-smooth").value = String(saved.options.smooth);
    $<HTMLInputElement>("ex-height").value = saved.options.height ? String(saved.options.height) : "";
    $<HTMLInputElement>("ex-saveas").checked = saved.saveAs;
    $("export-source").textContent = result.ply;
    syncMesh();
    syncOutputs();
    dialog.showModal();
  });

  $("btn-export-go").addEventListener("click", async () => {
    const result = selectedResult();
    const image = selectedImage();
    if (!result || !image) return;
    const height = Number($<HTMLInputElement>("ex-height").value);
    const options: ExportOptions = {
      method: method(),
      resolution: Number($<HTMLSelectElement>("ex-resolution").value),
      threshold: Number($<HTMLInputElement>("ex-threshold").value),
      smooth: Number($<HTMLInputElement>("ex-smooth").value),
      faces: Number($<HTMLSelectElement>("ex-faces").value),
      height: height > 0 ? height : null,
    };
    const fmtName = format();
    const saveAs = $<HTMLInputElement>("ex-saveas").checked;
    localStorage.setItem(EXPORT_OPTIONS_KEY, JSON.stringify({ format: fmtName, options, saveAs }));
    Object.assign(saved, { format: fmtName, options, saveAs });
    const defaultPath = result.ply.replace(/\.ply$/i, "") + EXPORT_SUFFIX[fmtName];
    let outPath: string | null = null;
    if (saveAs) {
      const ext = EXPORT_SUFFIX[fmtName].split(".").pop()!;
      const picked = await saveDialog({ defaultPath, filters: [{ name: `.${ext}`, extensions: [ext] }] });
      if (!picked) return;
      outPath = picked;
    }
    dialog.close();
    jobs.push({
      id: "j-" + uid(), kind: "export", imageId: image.id, imageName: image.name, params: result.params,
      status: "queued", stage: "", step: 0, total: 0, message: "Queued",
      export: { ply: result.ply, format: fmtName, out_path: outPath, options },
    });
    renderQueue();
    switchDock("queue");
    pump();
  });
}

async function stopWorker() {
  await invoke("worker_stop");
}

// ---------------------------------------------------------------------------------------------------------------
// Settings dialog

const dialog = $<HTMLDialogElement>("settings");

async function refreshInfo() {
  info = await invoke<AppInfo>("app_info");
  $("app-version").textContent = "v" + info.version;
  const envStatus = $("env-status");
  envStatus.textContent = info.python_exists
    ? `Using ${info.python}`
    : `No Python environment yet. "Set up" creates one in ${info.env_dir} with CUDA PyTorch and Idol (about 5 GB).`;
  $("env-banner").classList.toggle("hidden", info.python_exists);
  $<HTMLInputElement>("cli-path").value = info.cli_path;
  $("cli-hint").textContent = info.cli_installed
    ? `Try: "${info.cli_path}" generate photo.jpg --mode real   ·   --help lists every option, --json gives machine-readable output.`
    : "This build has no separate CLI; the app takes the same commands: idol-studio.exe generate photo.jpg (pipe its output, e.g. | more).";
}

function fillSettingsForm() {
  $<HTMLInputElement>("set-python").value = settings.python_path;
  $<HTMLInputElement>("set-token").value = settings.hf_token;
  $<HTMLInputElement>("set-output").value = settings.output_dir;
  $<HTMLInputElement>("set-output").placeholder = info.output_dir;
  $<HTMLInputElement>("set-updates").checked = settings.check_updates;
}

async function saveSettingsFromForm() {
  settings.python_path = $<HTMLInputElement>("set-python").value.trim();
  settings.hf_token = $<HTMLInputElement>("set-token").value.trim();
  settings.output_dir = $<HTMLInputElement>("set-output").value.trim();
  settings.check_updates = $<HTMLInputElement>("set-updates").checked;
  await invoke("save_settings", { newSettings: settings });
  await refreshInfo();
  $("settings-saved").textContent = "Saved.";
  setTimeout(() => ($("settings-saved").textContent = ""), 2000);
}

function bindSettings() {
  const openSettings = async () => {
    await refreshInfo();
    fillSettingsForm();
    dialog.showModal();
  };
  $("btn-settings").addEventListener("click", openSettings);
  $("btn-env-open").addEventListener("click", openSettings);
  $("btn-save-settings").addEventListener("click", saveSettingsFromForm);
  dialog.querySelectorAll<HTMLAnchorElement>("a[data-url]").forEach((a) =>
    a.addEventListener("click", (e) => {
      e.preventDefault();
      openUrl(a.dataset.url!);
    }),
  );
  $("btn-token-show").addEventListener("click", () => {
    const input = $<HTMLInputElement>("set-token");
    input.type = input.type === "password" ? "text" : "password";
    $("btn-token-show").textContent = input.type === "password" ? "Show" : "Hide";
  });
  $("btn-python-browse").addEventListener("click", async () => {
    const file = await openDialog({ filters: [{ name: "Python", extensions: ["exe"] }] });
    if (typeof file === "string") $<HTMLInputElement>("set-python").value = file;
  });
  $("btn-output-browse").addEventListener("click", async () => {
    const dir = await openDialog({ directory: true });
    if (typeof dir === "string") $<HTMLInputElement>("set-output").value = dir;
  });
  const showPathStatus = (s: { dir: string; on_path: boolean; cli_exists: boolean }) => {
    const button = $<HTMLButtonElement>("btn-cli-path");
    button.disabled = !s.cli_exists;
    button.textContent = s.on_path ? "Remove from PATH" : "Add to PATH";
    $("cli-path-status").textContent = !s.cli_exists
      ? "Install Idol Studio with the installer to get idol-studio-cli on PATH."
      : s.on_path
        ? "On your PATH: type idol-studio-cli in any new terminal."
        : "Not on your PATH.";
  };
  const refreshPath = () => invoke<{ dir: string; on_path: boolean; cli_exists: boolean }>("cli_path_status").then(showPathStatus).catch(() => {});
  $("btn-settings").addEventListener("click", refreshPath);
  $("btn-cli-path").addEventListener("click", async () => {
    const on = $("btn-cli-path").textContent === "Add to PATH";
    try {
      showPathStatus(await invoke("cli_path_set", { on }));
      toast(on ? "Added to PATH. New terminals can run idol-studio-cli." : "Removed from PATH.");
    } catch (e) {
      toast(String(e), true);
    }
  });
  $("btn-cli-copy").addEventListener("click", async () => {
    await navigator.clipboard.writeText(`"${info.cli_path}"`);
    toast("CLI path copied.");
  });
  $("btn-output-open").addEventListener("click", () => openPath(info.output_dir).catch((e) => toast(String(e), true)));

  const setupLog = $("setup-log");
  $("btn-setup-env").addEventListener("click", async () => {
    await saveSettingsFromForm();
    await stopWorker(); // the venv can't be updated while Python has files open
    setupLog.textContent = "";
    setupLog.classList.remove("hidden");
    $<HTMLButtonElement>("btn-setup-env").disabled = true;
    try {
      await invoke("setup_environment");
    } catch (error) {
      setupLog.textContent += String(error) + "\n";
      $<HTMLButtonElement>("btn-setup-env").disabled = false;
    }
  });
  listen<string>("setup-log", (e) => {
    setupLog.textContent += e.payload + "\n";
    setupLog.scrollTop = setupLog.scrollHeight;
  });
  listen<{ ok: boolean; message: string }>("setup-done", async (e) => {
    setupLog.textContent += (e.payload.ok ? "✔ " : "✖ ") + e.payload.message + "\n";
    setupLog.scrollTop = setupLog.scrollHeight;
    $<HTMLButtonElement>("btn-setup-env").disabled = false;
    await refreshInfo();
    toast(e.payload.message, !e.payload.ok);
  });

  $("btn-test-env").addEventListener("click", async () => {
    await saveSettingsFromForm();
    const status = $("env-status");
    status.textContent = "Starting the worker...";
    try {
      const hello = await ask("hello", {}, "hello");
      status.textContent = hello.import_error
        ? `Python ${hello.python} runs, but Idol can't be imported: ${hello.import_error}`
        : `Python ${hello.python} · PyTorch ${hello.torch} · CUDA ${hello.cuda ? "yes" : "NO"}${hello.gpu ? ` · ${hello.gpu} (${hello.vram_gb} GB)` : ""} · Idol ${hello.idol} · Triton ${hello.triton ? "yes" : "no"}`;
    } catch (error) {
      status.textContent = String(error);
    }
  });

  $("btn-check-access").addEventListener("click", async () => {
    await saveSettingsFromForm();
    const list = $("access-list");
    list.innerHTML = `<li class="muted">Checking...</li>`;
    try {
      const result = await ask("check_access", { weights: settings.weights }, "access");
      list.innerHTML = "";
      if (!result.logged_in) list.innerHTML = `<li class="bad">No Hugging Face token found. Paste one above (Save first) or run "hf auth login".</li>`;
      for (const repo of result.repos as { repo: string; ok: boolean; reason?: string }[]) {
        const li = document.createElement("li");
        li.className = repo.ok ? "ok" : "bad";
        li.innerHTML = `${repo.ok ? "✔" : "✖"} <a href="#">${escapeHtml(repo.repo)}</a> ${repo.ok ? "" : `<span class="muted">(${repo.reason === "GatedRepoError" ? "accept the terms on this page" : repo.reason})</span>`}`;
        li.querySelector("a")!.addEventListener("click", (e) => {
          e.preventDefault();
          openUrl(`https://huggingface.co/${repo.repo}`);
        });
        list.appendChild(li);
      }
    } catch (error) {
      list.innerHTML = `<li class="bad">${escapeHtml(String(error))}</li>`;
    }
  });
}

// ---------------------------------------------------------------------------------------------------------------
// Updates

function showUpdate(update: UpdateInfo) {
  pendingUpdate = update;
  $("update-text").textContent = `Idol Studio ${update.latest} is available (you have ${update.current}).`;
  $("update-banner").classList.remove("hidden");
  $<HTMLButtonElement>("btn-install-update").classList.toggle("hidden", !update.asset_url);
  $("btn-check-updates").textContent = "Update available";
  $("btn-check-updates").classList.add("attention");
}

function describeCheck(update: UpdateInfo) {
  const when = new Date(update.checked_at * 1000).toLocaleTimeString();
  if (update.available) return `Version ${update.latest} is available (checked ${when}).`;
  if (!update.latest) return `No releases published yet (checked ${when}).`;
  return `You have the latest version, ${update.current} (checked ${when}).`;
}

async function manualUpdateCheck() {
  $("update-status").textContent = "Checking...";
  try {
    const update = await invoke<UpdateInfo>("check_for_updates");
    $("update-status").textContent = describeCheck(update);
    if (!update.available) toast(describeCheck(update));
  } catch (error) {
    $("update-status").textContent = String(error);
    toast(`Update check failed: ${error}`, true);
  }
}

function bindUpdates() {
  listen<UpdateInfo>("update-available", (e) => showUpdate(e.payload));
  listen<UpdateInfo>("update-checked", (e) => ($("update-status").textContent = describeCheck(e.payload)));
  listen<string>("update-error", (e) => log(`[update] check failed: ${e.payload}`));
  listen<{ received: number; total: number | null }>("update-progress", (e) => {
    const { received, total } = e.payload;
    $("update-text").textContent = `Downloading update... ${total ? Math.round((received / total) * 100) + "%" : (received / 2 ** 20).toFixed(1) + " MB"}`;
  });
  $("btn-check-updates").addEventListener("click", () => (pendingUpdate ? $("update-banner").classList.remove("hidden") : manualUpdateCheck()));
  $("btn-check-updates-2").addEventListener("click", manualUpdateCheck);
  $("btn-update-later").addEventListener("click", () => $("update-banner").classList.add("hidden"));
  $("btn-update-notes").addEventListener("click", () => pendingUpdate && openUrl(pendingUpdate.page_url));
  $("btn-install-update").addEventListener("click", async () => {
    if (!pendingUpdate?.asset_url) return;
    if (jobs.some((j) => j.status === "running") && !confirm("A generation is running. Install the update anyway?")) return;
    $<HTMLButtonElement>("btn-install-update").disabled = true;
    try {
      await invoke("install_update", { url: pendingUpdate.asset_url, name: pendingUpdate.asset_name ?? "IdolStudio-setup.exe" });
    } catch (error) {
      toast(`Update failed: ${error}`, true);
      $<HTMLButtonElement>("btn-install-update").disabled = false;
    }
  });
}

// ---------------------------------------------------------------------------------------------------------------
// Toasts

function toast(message: string, error = false) {
  const el = document.createElement("div");
  el.className = "toast" + (error ? " error" : "");
  el.textContent = message;
  document.body.appendChild(el);
  setTimeout(() => el.classList.add("show"), 10);
  setTimeout(() => {
    el.classList.remove("show");
    setTimeout(() => el.remove(), 300);
  }, error ? 7000 : 3000);
}

// ---------------------------------------------------------------------------------------------------------------
// Start

async function main() {
  settings = await invoke<Settings>("get_settings");
  await refreshInfo();
  syncOptionsToForm();
  bindForm();
  bindChrome();
  bindSettings();
  bindUpdates();

  await listen<WorkerEvent>("worker-event", (e) => onWorkerEvent(e.payload));
  await listen<string>("worker-log", (e) => log(e.payload));
  await listen<string>("worker-log-replace", (e) => log(e.payload, true));
  await listen<number | null>("worker-exit", (e) => {
    log(`[worker] exited${e.payload !== null ? ` (code ${e.payload})` : ""}`);
    modelLoaded = false;
    setWorkerState("stopped");
    for (const job of jobs.filter((j) => j.status === "running")) {
      job.status = "error";
      job.message = "The worker stopped. See the Log tab.";
    }
    for (const [id, settle] of pendingRequests) {
      settle({ event: "error", id, message: "The worker stopped. See the Log tab." });
    }
    pendingRequests.clear();
    renderQueue();
    pump();
  });

  // The library moved from localStorage to library.json (shared with the CLI) in 0.2.0: import it once.
  const old = loadJson<LibraryImage[] | null>(LIBRARY_KEY, null);
  if (old) {
    await invoke("library_import", { library: old }).catch((e) => log(`[library] ${e}`));
    localStorage.removeItem(LIBRARY_KEY);
  }
  library = await invoke<LibraryImage[]>("library_load");
  selectedImageId = library[0]?.id ?? null;
  // Results added by the CLI (or another window) show up right away.
  await listen("library-changed", async () => {
    library = await invoke<LibraryImage[]>("library_load");
    if (!library.some((i) => i.id === selectedImageId)) selectedImageId = library[0]?.id ?? null;
    const image = selectedImage();
    const newest = image?.results[0];
    if (newest && !image!.results.some((r) => r.id === selectedResultId)) {
      selectImage(image!.id);
    } else if (newest && newest.id !== selectedResultId && newest.created > Date.now() - 60_000) {
      selectedResultId = newest.id; // a fresh result for the image on screen: show it
      renderLibrary();
      renderResults();
      showResult(newest);
    } else {
      renderLibrary();
      renderResults();
    }
  });

  renderLibrary();
  renderQueue();
  selectImage(selectedImageId);
  switchTab("3d");

  // Start the worker in the background so the first generation doesn't wait for Python + PyTorch to import.
  if (info.python_exists) ask("hello", {}, "hello").catch((e) => log(`[worker] ${e}`));
}

main().catch((error) => {
  console.error(error);
  toast(String(error), true);
});
