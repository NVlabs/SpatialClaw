import argparse
from collections import deque
import datetime
import json
import os
import random
import re
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
import urllib.error
from typing import Any, Dict, List, Optional, Tuple
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


class RegistryError(RuntimeError):
    """The shared endpoint registry cannot be read or safely published."""


_EPHEMERAL_PORT_RANGE_PATH = "/proc/sys/net/ipv4/ip_local_port_range"
_FALLBACK_EPHEMERAL_PORT_RANGE = (32768, 60999)
_AUTO_PORT_ATTEMPTS = 3
_EADDRINUSE_MARKER = "OSError: [Errno 98] Address already in use"


class _StderrCapture:
    """Bounded diagnostics plus a durable exact bind-collision signal."""

    def __init__(self) -> None:
        self.tail: deque[str] = deque(maxlen=256)
        self.bind_collision = threading.Event()
        self.forward_error: Optional[BaseException] = None


def _read_registry(path: str) -> Dict[str, Any]:
    """Read a registry without treating corrupt content as an empty registry."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            raw = f.read()
    except FileNotFoundError:
        return {}

    # A brand-new zero-byte file is equivalent to no registry. Any nonempty
    # invalid content is evidence that publication is unsafe, so fail closed.
    if raw == '':
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RegistryError(
            f'Refusing to overwrite corrupt nonempty registry: {path}'
        ) from exc
    if not isinstance(data, dict):
        raise RegistryError(
            f'Refusing to overwrite registry with non-object root: {path}'
        )
    return data


def _atomic_write_registry(path: str, data: Dict[str, Any]) -> None:
    """Durably replace ``path`` using a fsynced same-directory tempfile."""
    if not isinstance(data, dict):
        raise TypeError('Registry root must be a JSON object')

    absolute_path = os.path.abspath(path)
    directory = os.path.dirname(absolute_path)
    os.makedirs(directory, exist_ok=True)
    fd, temporary_path = tempfile.mkstemp(
        prefix=f'.{os.path.basename(path)}.', suffix='.tmp', dir=directory,
    )
    try:
        try:
            mode = os.stat(absolute_path).st_mode & 0o777
        except FileNotFoundError:
            mode = 0o600
        os.fchmod(fd, mode)
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            fd = -1
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.write('\n')
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary_path, absolute_path)
        temporary_path = ''

        # Make the directory entry durable too. Publication must not be
        # reported successful if the filesystem cannot persist the rename.
        directory_fd = os.open(
            directory, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0)
        )
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if fd >= 0:
            os.close(fd)
        if temporary_path:
            try:
                os.unlink(temporary_path)
            except FileNotFoundError:
                pass


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


def _ephemeral_port_range(
    path: str = _EPHEMERAL_PORT_RANGE_PATH,
) -> Tuple[int, int]:
    """Return the node's outbound ephemeral range, with a safe fallback."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            values = [int(value) for value in f.read().split()]
        if len(values) != 2 or not 1 <= values[0] <= values[1] <= 65535:
            raise ValueError('invalid ephemeral port range')
        return values[0], values[1]
    except (OSError, TypeError, ValueError):
        return _FALLBACK_EPHEMERAL_PORT_RANGE


def find_free_port(
    min_port=30001,
    max_port=65535,
    *,
    excluded_ports: Optional[set[int]] = None,
) -> int:
    if min_port > max_port:
        raise ValueError('min_port must be less than or equal to max_port')

    ephemeral_start, ephemeral_end = _ephemeral_port_range()
    excluded = set(excluded_ports or ())
    ports_to_try = [
        port for port in range(min_port, max_port + 1)
        if not ephemeral_start <= port <= ephemeral_end
        and port not in excluded
    ]
    random.shuffle(ports_to_try)

    for port in ports_to_try:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('0.0.0.0', port))
                return port
        except OSError:
            continue

    raise IOError(
        "No available non-ephemeral port found within the range "
        f"[{min_port}, {max_port}]"
    )


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

    model_revision = getattr(args, 'revision', None)
    if model_revision:
        vllm_args.extend(['--revision', model_revision])

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

    if 'cosmos3' in model_id:
        vllm_args.extend([
            '--mm-encoder-tp-mode', 'data',
            '--async-scheduling',
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
    *,
    synthetic_keepalive_requested: bool = False,
) -> None:
    with FileLock(lock_file):
        serve_dict = _read_registry(serve_file)
        
        model_key = args.model if args.served_model_name is None \
            else args.served_model_name

        if model_key not in serve_dict:
            serve_dict[model_key] = {}
        elif not isinstance(serve_dict[model_key], dict):
            raise RegistryError(
                f'Registry bucket for model {model_key!r} is not an object'
            )

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
            # Additive metadata: old clients ignore these keys.  Managed
            # clients use them to avoid absorbing an unrelated pool that
            # happens to expose the same served model name.
            'pool_id': os.environ.get('SPATIAL_AGENT_SERVICE_POOL_ID'),
            'namespace': os.environ.get('SPATIAL_AGENT_SERVICE_NAMESPACE'),
            'slot_id': os.environ.get('SPATIAL_AGENT_SERVICE_SLOT_ID'),
            'generation': os.environ.get('SPATIAL_AGENT_SERVICE_GENERATION'),
            'server_spec_digest': os.environ.get(
                'SPATIAL_AGENT_SERVER_SPEC_DIGEST'
            ),
            'topology_digest': os.environ.get(
                'SPATIAL_AGENT_TOPOLOGY_DIGEST'
            ),
            'state': 'starting',
            'accepting': False,
            'draining': False,
            'backend': 'vllm',
            'protocol': 'openai-v1',
            'requested_model': args.model,
            'requested_model_revision': getattr(args, 'revision', None),
            'effective_model': model_key,
            'kv_cache_dtype': args.kv_cache_dtype,
            'quantization': args.quantization,
            'gpu_identity': {
                'selected_devices': [str(gpu) for gpu in gpus],
                'cuda_visible_devices': [str(gpu) for gpu in gpus],
                'tensor_parallel_size': int(args.tp),
                'hostname': socket.gethostname(),
            },
            'synthetic_keepalive_requested': bool(
                synthetic_keepalive_requested
            ),
            'synthetic_keepalive_effective': False,
            'health': {
                'status': 'starting',
                'backend': 'vllm',
                'protocol': 'openai-v1',
                'effective_model': model_key,
                'requested_model_revision': getattr(args, 'revision', None),
            },
        }

        _atomic_write_registry(serve_file, serve_dict)


def update_record_state(
    serve_file: str,
    lock_file: str,
    model: str,
    uid: str,
    *,
    state: str,
    accepting: bool,
    draining: bool = False,
    health: Optional[Dict[str, Any]] = None,
    synthetic_keepalive_requested: Optional[bool] = None,
    synthetic_keepalive_effective: Optional[bool] = None,
) -> None:
    """Durably publish endpoint lifecycle state after a health transition."""
    with FileLock(lock_file):
        serve_dict = _read_registry(serve_file)
        model_records = serve_dict.get(model)
        if not isinstance(model_records, dict):
            raise RegistryError(
                f'Cannot publish readiness: model bucket {model!r} is missing'
            )
        record = model_records.get(uid)
        if not isinstance(record, dict):
            raise RegistryError(
                f'Cannot publish readiness: endpoint record {uid!r} is missing'
            )
        record.update({
            'state': state,
            'accepting': bool(accepting),
            'draining': bool(draining),
            'state_change_time': get_current_time(),
        })
        if health is not None:
            record['health'] = dict(health)
        if synthetic_keepalive_requested is not None:
            record['synthetic_keepalive_requested'] = bool(
                synthetic_keepalive_requested
            )
        if synthetic_keepalive_effective is not None:
            record['synthetic_keepalive_effective'] = bool(
                synthetic_keepalive_effective
            )
        _atomic_write_registry(serve_file, serve_dict)


def _process_owns_listening_port(
    pid: int,
    port: int,
    *,
    proc_root: str = '/proc',
) -> bool:
    """Prove that ``pid`` owns a TCP LISTEN socket for ``port`` on Linux."""
    process_root = os.path.join(proc_root, str(pid))
    socket_inodes: set[str] = set()
    try:
        with os.scandir(os.path.join(process_root, 'fd')) as entries:
            for entry in entries:
                try:
                    target = os.readlink(entry.path)
                except OSError:
                    continue
                match = re.fullmatch(r'socket:\[([0-9]+)\]', target)
                if match:
                    socket_inodes.add(match.group(1))
    except OSError:
        return False
    if not socket_inodes:
        return False

    for table_name in ('tcp', 'tcp6'):
        try:
            with open(
                os.path.join(process_root, 'net', table_name),
                'r',
                encoding='ascii',
                errors='replace',
            ) as table:
                next(table, None)
                for line in table:
                    fields = line.split()
                    if len(fields) < 10 or fields[3] != '0A':
                        continue
                    try:
                        local_port = int(fields[1].rsplit(':', 1)[1], 16)
                    except (IndexError, ValueError):
                        continue
                    if local_port == port and fields[9] in socket_inodes:
                        return True
        except OSError:
            continue
    return False


def wait_until_ready(
    port: int,
    process: subprocess.Popen,
    timeout: float = 1800.0,
    *,
    expected_model: Optional[str] = None,
    owner_pid: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """Return verified vLLM health identity, or ``None`` on readiness failure."""
    deadline = time.monotonic() + timeout
    url = f"http://127.0.0.1:{port}/v1/models"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            return None
        try:
            request = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(request, timeout=3) as response:
                if 200 <= int(response.status) < 300:
                    payload = json.loads(response.read().decode('utf-8'))
                    entries = payload.get('data', [])
                    served_models = sorted({
                        str(entry['id'])
                        for entry in entries
                        if isinstance(entry, dict) and entry.get('id')
                    })
                    if expected_model and expected_model not in served_models:
                        time.sleep(2)
                        continue
                    if not served_models:
                        time.sleep(2)
                        continue
                    if owner_pid is not None and not _process_owns_listening_port(
                        owner_pid, port,
                    ):
                        # A matching foreign server can win the same-port race.
                        # Never publish its health under this process identity.
                        time.sleep(2)
                        continue
                    return {
                        'status': 'ok',
                        'backend': 'vllm',
                        'protocol': 'openai-v1',
                        'effective_model': expected_model or served_models[0],
                        'served_models': served_models,
                    }
        except Exception:
            pass
        time.sleep(2)
    return None


def _service_readiness_timeout() -> float:
    """Return the managed startup bound without changing legacy behavior."""
    managed = bool(os.environ.get("SPATIAL_AGENT_SERVICE_POOL_ID", "").strip())
    raw = os.environ.get("SPATIAL_AGENT_SERVICE_STARTUP_TIMEOUT_SECONDS")
    if raw is None:
        # Historical unmanaged services were not killed after 30 minutes;
        # preserve their effective four-hour job-wall startup window. Managed
        # chains export a shorter explicit bound and own safe teardown.
        return 1800.0 if managed else 4 * 3600.0
    try:
        timeout = float(raw)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "SPATIAL_AGENT_SERVICE_STARTUP_TIMEOUT_SECONDS must be numeric"
        ) from exc
    upper_bound = 1800.0 if managed else 4 * 3600.0
    if not 1.0 <= timeout <= upper_bound:
        raise RuntimeError(
            "SPATIAL_AGENT_SERVICE_STARTUP_TIMEOUT_SECONDS must be between "
            f"1 and {int(upper_bound)} seconds"
        )
    return timeout


def terminate_process(process: subprocess.Popen, timeout: float = 30.0) -> None:
    """Best-effort bounded cleanup after a failed readiness check."""
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def _spawn_with_stderr_tee(
    launcher: List[str],
    envs: Dict[str, str],
    *,
    stdout,
    stderr_sink,
) -> Tuple[subprocess.Popen, _StderrCapture, threading.Thread]:
    """Start vLLM while preserving a bounded, live copy of child stderr."""
    process = subprocess.Popen(
        launcher,
        stdout=stdout,
        stderr=subprocess.PIPE,
        env=envs,
        text=True,
        encoding='utf-8',
        errors='replace',
        bufsize=1,
    )
    capture = _StderrCapture()
    sink = stderr_sink if stderr_sink is not None else sys.stderr
    stream = process.stderr
    if stream is None:
        terminate_process(process)
        raise RuntimeError('vLLM stderr pipe was not created')

    def _forward_stderr() -> None:
        sink_available = True
        for line in stream:
            capture.tail.append(line)
            if _EADDRINUSE_MARKER in line:
                capture.bind_collision.set()
            if sink_available:
                try:
                    print(line, end='', file=sink, flush=True)
                except BaseException as exc:
                    # Keep draining the pipe so a logging failure cannot block
                    # the child. The launcher retains the bounded diagnostics.
                    capture.forward_error = exc
                    sink_available = False

    thread = threading.Thread(
        target=_forward_stderr,
        daemon=True,
        name='vllm-stderr-tee',
    )
    try:
        thread.start()
    except BaseException:
        stream.close()
        terminate_process(process)
        raise
    return process, capture, thread


def _launch_until_ready(
    args: argparse.Namespace,
    envs: Dict[str, str],
    model_key: str,
    *,
    stdout,
    stderr_sink,
) -> Tuple[subprocess.Popen, Dict[str, Any], threading.Thread]:
    """Start one healthy vLLM generation, retrying only auto-port collisions."""
    auto_port = args.port is None
    attempts = _AUTO_PORT_ATTEMPTS if auto_port else 1
    attempted_ports: set[int] = set()
    startup_timeout = _service_readiness_timeout()
    startup_deadline = time.monotonic() + startup_timeout

    for attempt in range(1, attempts + 1):
        if startup_deadline - time.monotonic() <= 0:
            raise RuntimeError(
                'vLLM server startup exhausted its configured readiness timeout'
            )
        if auto_port:
            args.port = find_free_port(excluded_ports=attempted_ports)
            print(
                '[Launcher] No port specified. Found and using free port: '
                f'{args.port}',
                flush=True,
            )
        port = int(args.port)
        attempted_ports.add(port)
        launcher = get_launcher(args)
        process, stderr_capture, stderr_thread = _spawn_with_stderr_tee(
            launcher,
            envs,
            stdout=stdout,
            stderr_sink=stderr_sink,
        )
        print(
            f'[Launcher] vLLM server (PID: {process.pid}) for model '
            f'"{model_key}" started (port attempt {attempt}/{attempts}).',
            flush=True,
        )

        remaining = startup_deadline - time.monotonic()
        if remaining <= 0:
            terminate_process(process)
            stderr_thread.join(timeout=5.0)
            raise RuntimeError(
                'vLLM server startup exhausted its configured readiness timeout'
            )
        readiness = wait_until_ready(
            port,
            process,
            timeout=remaining,
            expected_model=model_key,
            owner_pid=process.pid,
        )
        if readiness is not None and process.poll() is None:
            return process, readiness, stderr_thread

        terminate_process(process)
        stderr_thread.join(timeout=5.0)
        bind_collision = stderr_capture.bind_collision.is_set()
        if auto_port and bind_collision and attempt < attempts:
            print(
                '[Launcher] vLLM port bind collided on '
                f'{port}; retrying inside the same job with a fresh port '
                f'({attempt + 1}/{attempts}).',
                file=stderr_sink if stderr_sink is not None else sys.stderr,
                flush=True,
            )
            args.port = None
            continue
        if bind_collision:
            raise RuntimeError(
                f'vLLM server could not bind port {port}; '
                f'port retry attempts exhausted ({attempt}/{attempts})'
            )
        raise RuntimeError(f'vLLM server failed readiness on port {port}')

    raise RuntimeError('vLLM server startup exhausted its bounded attempts')


def _synthetic_keepalive_enabled() -> bool:
    """Preserve legacy availability unless managed lifecycle disables it.

    Managed manager scripts explicitly export ``0`` after sourcing private
    environment files.  An unset variable therefore retains the historical
    unmanaged behavior until the matched lifecycle pilot passes, preventing a
    staged rollout from changing official legacy results through idle reaping.
    """
    raw = os.environ.get("SPATIAL_AGENT_ENABLE_SYNTHETIC_KEEPALIVE")
    if raw is None:
        return True
    return raw.strip().lower() in {"1", "true", "yes"}


def _ready_health_payload(
    args: argparse.Namespace,
    selected_gpus: List[int],
    readiness: Dict[str, Any],
    *,
    synthetic_keepalive_requested: bool,
    synthetic_keepalive_effective: bool,
) -> Dict[str, Any]:
    """Build the attested identity stored with a READY endpoint."""
    model_key = args.model if args.served_model_name is None \
        else args.served_model_name
    return {
        **readiness,
        'requested_model': args.model,
        'requested_model_revision': getattr(args, 'revision', None),
        'effective_model': model_key,
        'tensor_parallel_size': int(args.tp),
        'max_model_len': int(args.max_model_len),
        'max_num_seqs': int(args.max_num_seqs),
        'kv_cache_dtype': args.kv_cache_dtype,
        'quantization': args.quantization,
        'gpu': {
            'selected_devices': [str(gpu) for gpu in selected_gpus],
            'cuda_visible_devices': [str(gpu) for gpu in selected_gpus],
            'hostname': socket.gethostname(),
        },
        'synthetic_keepalive': {
            'requested': bool(synthetic_keepalive_requested),
            'effective': bool(synthetic_keepalive_effective),
        },
    }


def cleanup_record(
    serve_file: str, 
    lock_file: str, 
    model: str, 
    uid: str
) -> None:
    with FileLock(lock_file):
        try:
            serve_dict = _read_registry(serve_file)
        except RegistryError as exc:
            # Never turn a corrupt nonempty registry into an empty one during
            # cleanup. An operator must reconcile the damaged registry.
            print(f'[Launcher] Cleanup failed closed: {exc}', flush=True)
            return

        model_records = serve_dict.get(model)
        if not isinstance(model_records, dict) or uid not in model_records:
            return
        del model_records[uid]
        if not model_records:
            del serve_dict[model]
        _atomic_write_registry(serve_file, serve_dict)


# ---------------------------------------------------------------------------
# GPU keepalive — prevent cluster idle-GPU job reaper (DCGM_FI_DEV_GPU_UTIL)
#
# When explicitly enabled, sends a completion request at the configured
# interval (60 seconds by default).
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
    _keepalive_stop.clear()

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

        envs, selected_gpus = prepare_envs(args.tp)

        process = None
        stderr_thread = None
        model_key = args.model if args.served_model_name is None \
            else args.served_model_name
        try:
            # Retry an auto-port EADDRINUSE inside this exact Slurm job. Failed
            # attempts remain private: no endpoint is published until one
            # process has passed the non-inference readiness check.
            process, readiness, stderr_thread = _launch_until_ready(
                args,
                envs,
                model_key,
                stdout=None,
                stderr_sink=None,
            )
            pid = process.pid

            # Publish only the successful process and its final port.
            keepalive_requested = _synthetic_keepalive_enabled()
            setup_record(
                serve_file, lock_file, args, uid, str(pid), selected_gpus,
                synthetic_keepalive_requested=keepalive_requested,
            )

            keepalive_effective = False
            if keepalive_requested:
                keepalive_thread = start_vllm_keepalive(args.port, model_key)
                keepalive_effective = keepalive_thread.is_alive()
                if not keepalive_effective:
                    terminate_process(process)
                    raise RuntimeError(
                        'Synthetic keepalive was requested but did not start'
                    )
            else:
                print('[Keepalive] Disabled (normal lifecycle mode)', flush=True)

            health = _ready_health_payload(
                args, selected_gpus, readiness,
                synthetic_keepalive_requested=keepalive_requested,
                synthetic_keepalive_effective=keepalive_effective,
            )
            update_record_state(
                serve_file, lock_file, model_key, uid,
                state='ready', accepting=True,
                health=health,
                synthetic_keepalive_requested=keepalive_requested,
                synthetic_keepalive_effective=keepalive_effective,
            )
            print(f'[Launcher] vLLM server READY on port {args.port}', flush=True)

            # 4. Wait for the process to complete
            returncode = process.wait()
            _keepalive_stop.set()
            print(f'[Launcher] vLLM server process completed with return code: {returncode}', flush=True)

        except BaseException:
            if process is not None:
                terminate_process(process)
            raise
        finally:
            _keepalive_stop.set()
            if stderr_thread is not None:
                stderr_thread.join(timeout=5.0)
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
                envs, selected_gpus = prepare_envs(args.tp)

                process = None
                stderr_thread = None
                model_key = args.model if args.served_model_name is None \
                    else args.served_model_name
                try:
                    process, readiness, stderr_thread = _launch_until_ready(
                        args,
                        envs,
                        model_key,
                        stdout=f,
                        stderr_sink=f,
                    )
                    pid = process.pid

                    # Publish only the successful process and its final port.
                    keepalive_requested = _synthetic_keepalive_enabled()
                    setup_record(
                        serve_file, lock_file, args, uid, str(pid), selected_gpus,
                        synthetic_keepalive_requested=keepalive_requested,
                    )

                    keepalive_effective = False
                    if keepalive_requested:
                        keepalive_thread = start_vllm_keepalive(
                            args.port, model_key
                        )
                        keepalive_effective = keepalive_thread.is_alive()
                        if not keepalive_effective:
                            terminate_process(process)
                            raise RuntimeError(
                                'Synthetic keepalive was requested but did not start'
                            )
                    else:
                        print('[Keepalive] Disabled (normal lifecycle mode)')

                    health = _ready_health_payload(
                        args, selected_gpus, readiness,
                        synthetic_keepalive_requested=keepalive_requested,
                        synthetic_keepalive_effective=keepalive_effective,
                    )
                    update_record_state(
                        serve_file, lock_file, model_key, uid,
                        state='ready', accepting=True,
                        health=health,
                        synthetic_keepalive_requested=keepalive_requested,
                        synthetic_keepalive_effective=keepalive_effective,
                    )
                    print(f'[Launcher] vLLM server READY on port {args.port}')

                    # 4. Wait for the process to complete
                    process.wait()
                    _keepalive_stop.set()

                except BaseException:
                    if process is not None:
                        terminate_process(process)
                    raise
                finally:
                    _keepalive_stop.set()
                    if stderr_thread is not None:
                        stderr_thread.join(timeout=5.0)
                    if process:
                        print(f'[Launcher] vLLM server (PID: {pid}) has terminated. Cleaning up record.')
                        cleanup_record(serve_file, lock_file, model_key, uid)
                    else:
                        print(f'[Launcher] Process failed to launch.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('vLLM Model Launcher')
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--revision', type=str, default=None)
    parser.add_argument('--served_model_name', type=str, default=None)
    parser.add_argument('--port', type=int, default=None)
    parser.add_argument('--tp', type=int, default=1)
    parser.add_argument('--max_model_len', type=int, default=65536)
    parser.add_argument('--max_num_seqs', type=int, default=16)

    parser.add_argument('--kv_cache_dtype', type=str, default='auto')
    parser.add_argument('--quantization', type=str, default='none')
    args = parser.parse_args()

    launch_vllm_server(args)
