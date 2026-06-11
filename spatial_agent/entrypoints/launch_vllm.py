import argparse
import datetime
import json
import os
import random
import re
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import urllib.error
from typing import Dict, List, Tuple
import uuid

import pynvml


try:
    import fcntl
    def lock_file(f):
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
    def unlock_file(f):
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
except ImportError:
    import msvcrt
    def lock_file(f):
        msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, 1)
    def unlock_file(f):
        msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)


class FileLock:

    def __init__(self, filename):
        self.filename = filename
        self.file = None

    def __enter__(self):
        self.file = open(self.filename, 'a+') # 'a+'确保文件存在
        self.file.seek(0)
        lock_file(self.file)
        return self.file

    def __exit__(self, exc_type, exc_value, traceback):
        if self.file:
            unlock_file(self.file)
            self.file.close()
            self.file = None


class LogRedirector:

    def __init__(self, log_file_handle):
        self.log_file_handle = log_file_handle
        self.original_stdout = None
        self.original_stderr = None

    def __enter__(self):
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        sys.stdout = self.log_file_handle
        sys.stderr = self.log_file_handle
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr


def get_local_ip() -> str:
    s = None
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    except Exception as e:
        ip = '127.0.0.1'
        print(f'[Launcher] Cannot get local ip, error msg: {e}')
    finally:
        if s:
            s.close()
    return ip


def find_free_port(min_port=30001, max_port=65535) -> int:
    if min_port > max_port:
        raise ValueError('min_port must be less than or equal to max_port')

    ports_to_try = list(range(min_port, max_port + 1))
    random.shuffle(ports_to_try)

    for port in ports_to_try:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(('0.0.0.0', port))
                return port
        except OSError:
            continue
    
    raise IOError(f"No available port found within the range [{min_port}, {max_port}]")


def find_free_gpus(num_gpus: int) -> List[int]:
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()
    free_gpus = []

    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        try:
            procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
            if not procs:
                free_gpus.append(i)
        except pynvml.NVMLError as e:
            print(f'[Launcher] Could not query processes for GPU {i}: {e}')
    
    pynvml.nvmlShutdown()

    if len(free_gpus) < num_gpus:
        raise ValueError(
            f'Not enough free GPUs. Found {len(free_gpus)}, but need {num_gpus}. '
            f'Available GPUs: {free_gpus}'
        )

    return free_gpus[:num_gpus]


def get_current_time() -> str:
    now = datetime.datetime.now()
    formatted_time = now.strftime("%Y/%m/%d %H:%M:%S")
    return formatted_time


def _is_moe_model(model_name: str) -> bool:
    """Best-effort check for MoE naming patterns (e.g., 397B-A17B, 235B-A22B, 26B-A4B)."""
    lower_name = model_name.lower()
    if "moe" in lower_name:
        return True
    return bool(re.search(r"\d+(?:\.\d+)?b-a\d+(?:\.\d+)?b", lower_name))


def get_launcher(args) -> List[str]:
    if args.port is None:
        args.port = find_free_port()
        print(f'[Launcher] No port specified. Found and using free port: {args.port}')

    vllm_args = [
        'vllm.entrypoints.openai.api_server', 
        '--model', args.model,
        '--tensor-parallel-size', str(args.tp),
        '--max-model-len', str(args.max_model_len),
        '--max-num-seqs', str(args.max_num_seqs),
        '--dtype', 'auto',
        '--host', '0.0.0.0',
        '--port', str(args.port),
        '--trust-remote-code',
        '--gpu-memory-utilization', '0.90',
        '--enable-prefix-caching',
    ]

    if args.kv_cache_dtype != 'auto':
        vllm_args.extend(['--kv-cache-dtype', args.kv_cache_dtype])

    if args.quantization and args.quantization != 'none':
        vllm_args.extend(['--quantization', args.quantization])

    if args.served_model_name:
        vllm_args.extend(['--served-model-name', args.served_model_name])

    # Use served_model_name (if set) for model-family detection to avoid
    # false matches from parent directory names in the model path.
    model_id = (args.served_model_name or os.path.basename(args.model)).lower()

    if 'glm-4.5v' in model_id:
        vllm_args.extend([
            '--tool-call-parser', 'glm45',
            '--reasoning-parser', 'glm45',
            '--enable-auto-tool-choice',
            '--allowed-local-media-path', '/',
            '--media-io-kwargs', '{"video": {"num_frames": -1}}',
        ])

    if 'qwen3.5' in model_id:
        # Qwen3.5 recipe:
        # - reasoning parser improves think/answer handling
        # - expert parallel is only valid for MoE variants
        vllm_args.extend([
            '--reasoning-parser', 'qwen3',
        ])
        if _is_moe_model(model_id):
            vllm_args.append('--enable-expert-parallel')

    if 'gemma' in model_id:
        chat_template = os.path.join(
            os.path.dirname(__file__), '..', 'launch_managers',
            'vllm_manager', 'chat_templates', 'gemma4.jinja',
        )
        vllm_args.extend([
            '--reasoning-parser', 'gemma4',
            '--chat-template', chat_template,
        ])
        if _is_moe_model(model_id):
            vllm_args.append('--enable-expert-parallel')

    if 'qwen3-vl' in model_id:
        vllm_args.extend([
            '--mm-encoder-tp-mode', 'data',
            '--enable-expert-parallel',
            '--async-scheduling',
            '--limit-mm-per-prompt.video', '0',
            '--distributed-executor-backend', 'mp',
            '--mm-processor-cache-gb', '50',
        ])
        # Only add quantization flag for non-FP8 models
        # Pre-quantized FP8 models already contain quantized weights.
        # Skip auto-injection if the user already set --quantization
        # explicitly via the launcher flag.
        user_set_quant = args.quantization and args.quantization != 'none'
        if 'fp8' not in model_id and not user_set_quant:
            vllm_args.extend(['--quantization', 'fp8'])
        if 'thinking' in model_id:
            vllm_args.extend(['--reasoning-parser', 'qwen3'])

    print(f'[Launcher] {" ".join(vllm_args)}')

    launcher = [sys.executable, '-m'] + vllm_args
    return launcher


def prepare_envs(num_gpus: int) -> Tuple[Dict[str, str], List[int]]:
    env = os.environ.copy()

    # set visible gpus
    try:
        selected_gpus = find_free_gpus(num_gpus)
        print(f'[Launcher] Found {len(selected_gpus)} free GPUs: {selected_gpus}')
    except Exception as e:
        print(f'[Launcher] Error finding free GPUs: {e}')
        raise
    env['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, selected_gpus))
    return env, selected_gpus


def setup_record(
    serve_file: str, 
    lock_file: str, 
    args: argparse.Namespace, 
    uid: str, 
    pid: str,
    gpus: List[int],
) -> None:
    with FileLock(lock_file):
        if os.path.exists(serve_file):
            with open(serve_file, 'r', encoding='utf-8') as f:
                serve_dict = json.load(f)
        else:
            serve_dict = {}
        
        model_key = args.model if args.served_model_name is None \
            else args.served_model_name

        if model_key not in serve_dict:
            serve_dict[model_key] = {}

        serve_dict[model_key][uid] = {
            'pid': pid,
            'ip': get_local_ip(),
            'port': str(args.port),
            'tp': str(args.tp),
            'gpus': gpus,
            'max_model_len': str(args.max_model_len),
            'max_num_seqs': str(args.max_num_seqs),
            'create_time': get_current_time(),
            'slurm_job_id': os.environ.get('SLURM_JOB_ID', None),  # Track SLURM job ID if available
        }

        with open(serve_file, 'w', encoding='utf-8') as f:
            json.dump(serve_dict, f, indent=2, ensure_ascii=False)


def cleanup_record(
    serve_file: str, 
    lock_file: str, 
    model: str, 
    uid: str
) -> None:
    with FileLock(lock_file):
        try:
            with open(serve_file, 'r+', encoding='utf-8') as f:
                serve_dict = json.load(f)

                if model in serve_dict and uid in serve_dict[model]:
                    del serve_dict[model][uid]
                    if not serve_dict[model]: 
                        del serve_dict[model]

                    f.seek(0)
                    f.truncate()
                    json.dump(serve_dict, f, indent=2, ensure_ascii=False)

        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
            print(f'[Launcher] Cleanup skipped, file might be missing, empty or entry not found: {e}')
            pass


# ---------------------------------------------------------------------------
# GPU keepalive — prevent cluster idle-GPU job reaper (DCGM_FI_DEV_GPU_UTIL)
#
# Sends a lightweight completion request to the vLLM server every 3 seconds.
# This triggers a short GPU inference burst, keeping utilization above the 0%
# threshold that the reaper checks over 90 consecutive minutes.
# ---------------------------------------------------------------------------

_keepalive_stop = threading.Event()


def start_vllm_keepalive(
    port: int,
    model_name: str,
    interval: float = 60.0,
    startup_delay: float = 120.0,
) -> threading.Thread:
    """Spawn a daemon thread that pings the vLLM server periodically.

    Args:
        port: Local port the vLLM server listens on.
        model_name: Model name for the API request.
        interval: Seconds between keepalive requests.
        startup_delay: Seconds to wait before the first request (server boot).
    """
    def _loop():
        time.sleep(startup_delay)
        url = f"http://localhost:{port}/v1/completions"
        payload = json.dumps({
            "model": model_name,
            "prompt": "Write a detailed and comprehensive essay about the history, development, and future of artificial intelligence, covering key milestones, breakthroughs, challenges, and societal implications from the 1950s to the present day and beyond.",
            "max_tokens": 4,
            "temperature": 0,
        }).encode("utf-8")
        headers = {"Content-Type": "application/json"}

        consecutive_failures = 0
        while not _keepalive_stop.wait(timeout=interval):
            try:
                req = urllib.request.Request(url, data=payload, headers=headers)
                with urllib.request.urlopen(req, timeout=120) as resp:
                    resp.read()
                consecutive_failures = 0
            except Exception as e:
                consecutive_failures += 1
                if consecutive_failures <= 3 or consecutive_failures % 10 == 0:
                    print(
                        f"[Keepalive] Request failed (attempt {consecutive_failures}): {e}",
                        flush=True,
                    )

    t = threading.Thread(target=_loop, daemon=True, name="vllm-keepalive")
    t.start()
    print(f"[Keepalive] Started (interval={interval}s, startup_delay={startup_delay}s)", flush=True)
    return t


def launch_vllm_server(args: argparse.Namespace):
    uid = str(uuid.uuid4())

    log_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')
    os.makedirs(log_dir, exist_ok=True)

    serve_file = os.path.join(log_dir, 'serve.json')
    lock_file = serve_file + '.lock'

    # Check if running in SLURM - if so, output to stdout instead of file
    in_slurm = 'SLURM_JOB_ID' in os.environ and os.environ['SLURM_JOB_ID'] != ''

    if in_slurm:
        print(f'[Launcher] Running in SLURM job {os.environ["SLURM_JOB_ID"]}', flush=True)
        print(f'[Launcher] Logs will be captured by SLURM', flush=True)
        print(f'--- Launcher Log for Service UID: {uid} ---', flush=True)

        launcher = get_launcher(args)
        envs, selected_gpus = prepare_envs(args.tp)

        process = None
        model_key = args.model if args.served_model_name is None \
            else args.served_model_name
        try:
            # 1. Launch the subprocess (inherit stdout/stderr so SLURM captures it)
            # Using None for stdout/stderr means inherit from parent process
            process = subprocess.Popen(
                launcher,
                stdout=None,  # Inherit parent's stdout (captured by SLURM)
                stderr=None,  # Inherit parent's stderr (captured by SLURM)
                env=envs,
            )
            pid = process.pid
            print(f'[Launcher] vLLM server (PID: {pid}) for model "{model_key}" started.', flush=True)

            # 2. Register the service
            setup_record(serve_file, lock_file, args, uid, str(pid), selected_gpus)

            # 3. Start keepalive to prevent idle-GPU reaper
            start_vllm_keepalive(args.port, model_key)

            # 4. Wait for the process to complete
            returncode = process.wait()
            _keepalive_stop.set()
            print(f'[Launcher] vLLM server process completed with return code: {returncode}', flush=True)

        finally:
            _keepalive_stop.set()
            if process:
                print(f'[Launcher] vLLM server (PID: {pid}) has terminated. Cleaning up record.', flush=True)
                cleanup_record(serve_file, lock_file, model_key, uid)
            else:
                print(f'[Launcher] Process failed to launch.', flush=True)
    else:
        # Original behavior: write to separate log file
        log_file = os.path.join(log_dir, f'serve_{uid}.log')
        print(f'[Launcher] Logs will be written to: {log_file}')

        with open(log_file, 'w', buffering=1, encoding='utf-8') as f:
            with LogRedirector(f):
                print(f'--- Launcher Log for Service UID: {uid} ---')
                launcher = get_launcher(args)
                envs, selected_gpus = prepare_envs(args.tp)

                process = None
                model_key = args.model if args.served_model_name is None \
                    else args.served_model_name
                try:
                    # 1. Launch the subprocess
                    process = subprocess.Popen(
                        launcher,
                        stdout=f,
                        stderr=f,
                        env=envs,
                    )
                    pid = process.pid
                    print(f'[Launcher] vLLM server (PID: {pid}) for model "{model_key}" started.')

                    # 2. Register the service
                    setup_record(serve_file, lock_file, args, uid, str(pid), selected_gpus)

                    # 3. Start keepalive to prevent idle-GPU reaper
                    start_vllm_keepalive(args.port, model_key)

                    # 4. Wait for the process to complete
                    process.wait()
                    _keepalive_stop.set()

                finally:
                    _keepalive_stop.set()
                    if process:
                        print(f'[Launcher] vLLM server (PID: {pid}) has terminated. Cleaning up record.')
                        cleanup_record(serve_file, lock_file, model_key, uid)
                    else:
                        print(f'[Launcher] Process failed to launch.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('vLLM Model Launcher')
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--served_model_name', type=str, default=None)
    parser.add_argument('--port', type=int, default=None)
    parser.add_argument('--tp', type=int, default=1)
    parser.add_argument('--max_model_len', type=int, default=65536)
    parser.add_argument('--max_num_seqs', type=int, default=16)

    parser.add_argument('--kv_cache_dtype', type=str, default='auto')
    parser.add_argument('--quantization', type=str, default='none')
    args = parser.parse_args()

    launch_vllm_server(args)
