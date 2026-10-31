"""Idol Studio worker.

Keeps one IdolPipeline loaded on the GPU and answers JSON-line requests on stdin. Replies and events go to
stdout as JSON lines; everything else (library prints, tqdm, warnings) is redirected to stderr, which the app
shows as the log.

Requests:  {"cmd": "hello" | "check_access" | "load" | "generate" | "preprocess" | "unload" | "cancel" | "quit", ...}
Events:    {"event": "hello" | "status" | "progress" | "done" | "preprocessed" | "error" | "access" | ..., ...}
"""
import json
import os
import queue
import sys
import threading
import time
import traceback

PROTOCOL = sys.stdout
sys.stdout = sys.stderr  # keep stdout clean for the protocol

_send_lock = threading.Lock()


def send(event, **data):
    with _send_lock:
        PROTOCOL.write(json.dumps({'event': event, **data}) + '\n')
        PROTOCOL.flush()


class Cancelled(Exception):
    pass


cancel_flag = threading.Event()
requests = queue.Queue()


def stdin_lines():
    """Yield stdin lines without ever leaving a blocking read pending on Windows.

    On Windows, a thread blocked in a synchronous read of a pipe also blocks other calls on that handle, and C
    runtime start-up in newly loaded DLLs queries the standard handles. So `import torch` (and every DLL loaded
    later) would hang while we wait for the next request. Peek first and only read what is already there."""
    if os.name != 'nt':
        yield from sys.stdin
        return
    import ctypes
    import msvcrt
    from ctypes import wintypes

    fd = sys.stdin.fileno()
    handle = msvcrt.get_osfhandle(fd)
    peek = ctypes.windll.kernel32.PeekNamedPipe
    peek.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD, ctypes.c_void_p,
                     ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p]
    peek.restype = wintypes.BOOL
    available = wintypes.DWORD()
    buffer = b''
    while True:
        if not peek(handle, None, 0, None, ctypes.byref(available), None):
            break  # pipe closed
        if available.value == 0:
            time.sleep(0.05)
            continue
        chunk = os.read(fd, available.value)
        if not chunk:
            break
        buffer += chunk
        while b'\n' in buffer:
            line, buffer = buffer.split(b'\n', 1)
            yield line.decode('utf-8', errors='replace')
    if buffer:
        yield buffer.decode('utf-8', errors='replace')


def read_stdin():
    """Read requests on a thread so 'cancel' takes effect while a generation is running."""
    for line in stdin_lines():
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            send('error', id=None, message=f'bad request: {line[:200]}')
            continue
        if request.get('cmd') == 'cancel':
            cancel_flag.set()
            continue
        requests.put(request)
    requests.put({'cmd': 'quit'})


# ---------------------------------------------------------------------------------------------------------------
# Lazy heavy imports, so 'hello' answers quickly and import errors are reported instead of crashing.

torch = None
idol = None


def import_idol():
    global torch, idol
    if idol is not None:
        return
    send('status', id=None, stage='import', message='Importing PyTorch and Idol...')
    import torch as _torch
    import idol as _idol
    import idol.sampler as sampler

    torch, idol = _torch, _idol

    # Report sampling progress (and honour cancel) by replacing the tqdm used by the sampler.
    def progress_range(iterable, desc='', disable=False, **_):
        items = list(iterable)
        total = len(items)
        started = time.time()
        for index, item in enumerate(items):
            if cancel_flag.is_set():
                raise Cancelled()
            yield item
            send('progress', id=CURRENT['id'], stage='sample', step=index + 1, total=total,
                 elapsed=round(time.time() - started, 2))

    sampler.tqdm = progress_range

    # torch.compile needs Triton (triton-windows on Windows). If compiling fails, run eagerly instead of crashing.
    try:
        import torch._dynamo
        torch._dynamo.config.suppress_errors = True
    except Exception:
        pass


def has_triton():
    try:
        import triton  # noqa: F401
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------------------------------------------

STATE = {'pipeline': None, 'config': None}
CURRENT = {'id': None}


def pipeline_config(request):
    return {
        'weights': request.get('weights') or 'speridlabs/idol',
        'dinov3_dtype': request.get('dinov3_dtype') or 'bf16',
        'optimize': bool(request.get('optimize', True)),
    }


def ensure_pipeline(request):
    import_idol()
    config = pipeline_config(request)
    if config['optimize'] and not has_triton():
        send('status', id=CURRENT['id'], stage='load',
             message='Triton not found: torch.compile is off (install triton-windows for faster sampling).')
        config['optimize'] = False
    if STATE['pipeline'] is not None and STATE['config'] == config:
        return STATE['pipeline']
    unload()
    send('status', id=CURRENT['id'], stage='load',
         message=f"Loading Idol ({config['weights']}, DINOv3 {config['dinov3_dtype']}, "
                 f"optimize {'on' if config['optimize'] else 'off'}). The first run downloads the weights...")
    started = time.time()
    from idol import IdolPipeline
    from idol.conditioning import EncoderAccessError
    try:
        pipeline = IdolPipeline.from_pretrained(config['weights'], dinov3_dtype=config['dinov3_dtype'],
                                                optimize=config['optimize'])
    except EncoderAccessError as error:
        raise UserError(str(error))
    STATE['pipeline'], STATE['config'] = pipeline, config
    BIREFNET.update(weights=None, model=None)  # the pipeline has its own copy
    send('loaded', id=CURRENT['id'], config=config, seconds=round(time.time() - started, 1), **gpu_memory())
    return pipeline


def unload():
    if STATE['pipeline'] is None:
        return
    STATE['pipeline'] = None
    STATE['config'] = None
    import gc
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
    send('unloaded', id=None)


def gpu_memory():
    if torch is None or not torch.cuda.is_available():
        return {}
    return {'vram_allocated_gb': round(torch.cuda.memory_allocated() / 2**30, 2),
            'vram_peak_gb': round(torch.cuda.max_memory_allocated() / 2**30, 2)}


class UserError(Exception):
    """An error with a message meant for the user (no traceback)."""


def open_image(path):
    from PIL import Image
    image = Image.open(path)
    image.load()
    return image


# ---------------------------------------------------------------------------------------------------------------
# Commands

def cmd_hello(request):
    info = {'python': sys.version.split()[0], 'executable': sys.executable,
            'hf_token': bool(os.environ.get('HF_TOKEN')), 'triton': has_triton()}
    try:
        import_idol()
        info['torch'] = torch.__version__
        info['cuda'] = torch.cuda.is_available()
        if info['cuda']:
            props = torch.cuda.get_device_properties(0)
            info['gpu'] = props.name
            info['vram_gb'] = round(props.total_memory / 2**30, 1)
        from importlib.metadata import version
        info['idol'] = version('idol')
    except Exception as error:
        info['import_error'] = f'{type(error).__name__}: {error}'
    send('hello', id=request.get('id'), **info)


def cmd_check_access(request):
    from huggingface_hub import auth_check, get_token, hf_hub_download
    weights = request.get('weights') or 'speridlabs/idol'
    results = []
    if os.path.isdir(weights):
        config_path = os.path.join(weights, 'config.json')
    else:
        config_path = hf_hub_download(weights, 'config.json')
        results.append({'repo': weights, 'ok': True})
    with open(config_path) as f:
        encoders = json.load(f)['image_encoders']
    for repo in (encoders['dinov3'], encoders['flux2_vae']):
        try:
            auth_check(repo)
            results.append({'repo': repo, 'ok': True})
        except Exception as error:
            results.append({'repo': repo, 'ok': False, 'reason': type(error).__name__})
    send('access', id=request.get('id'), logged_in=get_token() is not None, repos=results)


def cmd_load(request):
    ensure_pipeline(request)


BIREFNET = {'weights': None, 'model': None, 'image_size': 1024}


def background_remover(weights):
    """BiRefNet on its own (no gated encoders, ~0.5 GB), for crop previews before the full model is loaded."""
    if BIREFNET['weights'] == weights:
        return BIREFNET['model'], BIREFNET['image_size']
    import_idol()
    from safetensors.torch import load_file
    from idol.models import BiRefNet
    send('status', id=CURRENT['id'], stage='load', message='Loading background removal (BiRefNet)...')
    if os.path.isdir(weights):
        folder = weights
    else:
        from huggingface_hub import snapshot_download
        folder = snapshot_download(weights, allow_patterns=['config.json', 'birefnet.safetensors'])
    with open(os.path.join(folder, 'config.json')) as f:
        image_size = json.load(f)['image_size']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = BiRefNet()
    model.load_state_dict(load_file(os.path.join(folder, 'birefnet.safetensors')), strict=True)
    model = model.eval().requires_grad_(False).to(device)
    if device == 'cuda':
        model.half()
    BIREFNET.update(weights=weights, model=model, image_size=image_size)
    return model, image_size


def cmd_preprocess(request):
    import_idol()
    from idol.preprocess import CROP_MARGIN, preprocess_image
    mode = request.get('mode', 'synthetic')
    margin = request.get('crop_margin')
    margin = margin if margin is not None else CROP_MARGIN[mode]
    config = pipeline_config(request)
    if STATE['pipeline'] is not None and STATE['config'] and STATE['config']['weights'] == config['weights']:
        remover, image_size = STATE['pipeline'].background_remover, STATE['pipeline'].config['image_size']
    else:
        remover, image_size = background_remover(config['weights'])
    send('status', id=request['id'], stage='preprocess', message='Removing background and cropping...')
    with torch.no_grad():
        prepared = preprocess_image(open_image(request['image']), background_remover=remover,
                                    image_size=image_size, crop_margin=margin)
    os.makedirs(os.path.dirname(request['out_path']), exist_ok=True)
    prepared.save(request['out_path'])
    send('preprocessed', id=request['id'], path=request['out_path'])


def cmd_generate(request):
    pipeline = ensure_pipeline(request)
    job_id = request['id']
    base = request['out_base']
    os.makedirs(os.path.dirname(base), exist_ok=True)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Stage events around the pipeline's own steps.
    original_preprocess, original_decode = pipeline.preprocess, pipeline.decode

    def preprocess(*args, **kwargs):
        send('status', id=job_id, stage='preprocess', message='Removing background and cropping...')
        result = original_preprocess(*args, **kwargs)
        send('status', id=job_id, stage='sample', message='Encoding image and sampling...')
        return result

    def decode(*args, **kwargs):
        send('status', id=job_id, stage='decode', message='Decoding Gaussians...')
        return original_decode(*args, **kwargs)

    pipeline.preprocess, pipeline.decode = preprocess, decode
    started = time.time()
    try:
        output = pipeline(open_image(request['image']), mode=request.get('mode', 'synthetic'),
                          seed=int(request.get('seed', 42)), steps=int(request.get('steps', 50)),
                          cfg_strength=float(request.get('cfg_strength', 3.0)),
                          rescale_t=float(request.get('rescale_t', 3.0)),
                          num_gaussians=int(request.get('num_gaussians', 262144)),
                          crop_margin=request.get('crop_margin'))
    finally:
        del pipeline.preprocess, pipeline.decode  # back to the class methods
    seconds = time.time() - started

    send('status', id=job_id, stage='save', message='Saving PLY...')
    ply, preprocessed, camera_path = f'{base}.ply', f'{base}_input.png', f'{base}_camera.json'
    output.gaussian.save_ply(ply)
    output.preprocessed_image.save(preprocessed)
    with open(camera_path, 'w') as f:
        json.dump(output.camera, f, indent=2)
    params = {key: request.get(key) for key in ('mode', 'crop_margin', 'seed', 'steps', 'cfg_strength', 'rescale_t',
                                                'num_gaussians')}
    send('done', id=job_id, image=request['image'], params=params, ply=ply, preprocessed=preprocessed,
         camera_path=camera_path, camera=output.camera,
         seconds=round(seconds, 2), num_gaussians=int(output.gaussian.xyz.shape[0]), **gpu_memory())


COMMANDS = {'hello': cmd_hello, 'check_access': cmd_check_access, 'load': cmd_load,
            'preprocess': cmd_preprocess, 'generate': cmd_generate}


def main():
    threading.Thread(target=read_stdin, daemon=True).start()
    send('ready', id=None, pid=os.getpid())
    while True:
        request = requests.get()
        cmd = request.get('cmd')
        if cmd == 'quit':
            break
        if cmd == 'unload':
            unload()
            continue
        handler = COMMANDS.get(cmd)
        if handler is None:
            send('error', id=request.get('id'), message=f'unknown command: {cmd}')
            continue
        CURRENT['id'] = request.get('id')
        cancel_flag.clear()
        try:
            handler(request)
        except Cancelled:
            send('cancelled', id=request.get('id'))
        except UserError as error:
            send('error', id=request.get('id'), message=str(error))
        except Exception as error:
            traceback.print_exc()
            message = f'{type(error).__name__}: {error}'
            if torch is not None and isinstance(error, torch.cuda.OutOfMemoryError):
                message = ('Out of GPU memory. Try fewer Gaussians, close other GPU apps, '
                           'or set DINOv3 precision to bf16.\n' + message)
            send('error', id=request.get('id'), message=message)
        finally:
            CURRENT['id'] = None


if __name__ == '__main__':
    main()
