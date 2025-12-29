import gradio as gr

import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
# Pin to a single GPU for stability in multi-GPU setups
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
# More aggressive allocator settings to reduce fragmentation / spikes
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:64,garbage_collection_threshold:0.8"

from datetime import datetime
import shutil
import cv2
from typing import *
import gc
import torch
import numpy as np
from PIL import Image
import base64
import io
import sys
import time
import ctypes
from ctypes import wintypes
import subprocess
import json
from pathlib import Path
import threading
from trellis2.modules.sparse import SparseTensor
from trellis2.pipelines import Trellis2ImageTo3DPipeline
from trellis2.renderers import EnvMap
from trellis2.representations import Mesh
from trellis2.utils import render_utils
import o_voxel


MAX_SEED = np.iinfo(np.int32).max
TMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tmp')
_SESSION_DIRS = {}
_SESSION_DIRS_LOCK = threading.Lock()
MODES = [
    {"name": "Normal", "icon": "assets/app/normal.png", "render_key": "normal"},
    {"name": "Clay render", "icon": "assets/app/clay.png", "render_key": "clay"},
    {"name": "Base color", "icon": "assets/app/basecolor.png", "render_key": "base_color"},
    {"name": "HDRI forest", "icon": "assets/app/hdri_forest.png", "render_key": "shaded_forest"},
    {"name": "HDRI sunset", "icon": "assets/app/hdri_sunset.png", "render_key": "shaded_sunset"},
    {"name": "HDRI courtyard", "icon": "assets/app/hdri_courtyard.png", "render_key": "shaded_courtyard"},
]

# Preview settings
NORMAL_PREVIEW_VIEWS = 8
FULL_PREVIEW_VIEWS = 8
PREVIEW_RESOLUTION = 512
DEFAULT_MODE = 3
DEFAULT_STEP = 0


css = """
/* Overwrite Gradio Default Style */
.stepper-wrapper {
    padding: 0;
}

.stepper-container {
    padding: 0;
    align-items: center;
}

.step-button {
    flex-direction: row;
}

.step-connector {
    transform: none;
}

.step-number {
    width: 16px;
    height: 16px;
}

.step-label {
    position: relative;
    bottom: 0;
}

.wrap.center.full {
    inset: 0;
    height: 100%;
}

.wrap.center.full.translucent {
    background: var(--block-background-fill);
}

.meta-text-center {
    display: block !important;
    position: absolute !important;
    top: unset !important;
    bottom: 0 !important;
    right: 0 !important;
    transform: unset !important;
}

/* Previewer */
.previewer-container {
    position: relative;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    width: 100%;
    height: 722px;
    margin: 0 auto;
    padding: 20px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
}

.previewer-container .tips-icon {
    position: absolute;
    right: 10px;
    top: 10px;
    z-index: 10;
    border-radius: 10px;
    color: #fff;
    background-color: var(--color-accent);
    padding: 3px 6px;
    user-select: none;
}

.previewer-container .tips-text {
    position: absolute;
    right: 10px;
    top: 50px;
    color: #fff;
    background-color: var(--color-accent);
    border-radius: 10px;
    padding: 6px;
    text-align: left;
    max-width: 300px;
    z-index: 10;
    transition: all 0.3s;
    opacity: 0%;
    user-select: none;
}

.previewer-container .tips-text p {
    font-size: 14px;
    line-height: 1.2;
}

.tips-icon:hover + .tips-text { 
    display: block;
    opacity: 100%;
}

/* Row 1: Display Modes */
.previewer-container .mode-row {
    width: 100%;
    display: flex;
    gap: 8px;
    justify-content: center;
    margin-bottom: 20px;
    flex-wrap: wrap;
}
.previewer-container .mode-btn {
    width: 24px;
    height: 24px;
    border-radius: 50%;
    cursor: pointer;
    opacity: 0.5;
    transition: all 0.2s;
    border: 2px solid #ddd;
    object-fit: cover;
}
.previewer-container .mode-btn:hover { opacity: 0.9; transform: scale(1.1); }
.previewer-container .mode-btn.active {
    opacity: 1;
    border-color: var(--color-accent);
    transform: scale(1.1);
}

/* Row 2: Display Image */
.previewer-container .display-row {
    margin-bottom: 20px;
    min-height: 400px;
    width: 100%;
    flex-grow: 1;
    display: flex;
    justify-content: center;
    align-items: center;
}
.previewer-container .previewer-main-image {
    max-width: 100%;
    max-height: 100%;
    flex-grow: 1;
    object-fit: contain;
    display: none;
}
.previewer-container .previewer-main-image.visible {
    display: block;
}

/* Row 3: Custom HTML Slider */
.previewer-container .slider-row {
    width: 100%;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 10px;
    padding: 0 10px;
}

.previewer-container input[type=range] {
    -webkit-appearance: none;
    width: 100%;
    max-width: 400px;
    background: transparent;
}
.previewer-container input[type=range]::-webkit-slider-runnable-track {
    width: 100%;
    height: 8px;
    cursor: pointer;
    background: #ddd;
    border-radius: 5px;
}
.previewer-container input[type=range]::-webkit-slider-thumb {
    height: 20px;
    width: 20px;
    border-radius: 50%;
    background: var(--color-accent);
    cursor: pointer;
    -webkit-appearance: none;
    margin-top: -6px;
    box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    transition: transform 0.1s;
}
.previewer-container input[type=range]::-webkit-slider-thumb:hover {
    transform: scale(1.2);
}

/* Overwrite Previewer Block Style */
.gradio-container .padded:has(.previewer-container) {
    padding: 0 !important;
}

.gradio-container:has(.previewer-container) [data-testid="block-label"] {
    position: absolute;
    top: 0;
    left: 0;
}
"""


head = """
<script>
    function refreshView(mode, step) {
        // 1. Find current mode and step
        const allImgs = document.querySelectorAll('.previewer-main-image');
        for (let i = 0; i < allImgs.length; i++) {
            const img = allImgs[i];
            if (img.classList.contains('visible')) {
                const id = img.id;
                const [_, m, s] = id.split('-');
                if (mode === -1) mode = parseInt(m.slice(1));
                if (step === -1) step = parseInt(s.slice(1));
                break;
            }
        }

        // 2. Hide ALL images
        allImgs.forEach(img => img.classList.remove('visible'));

        // 3. Construct the specific ID for the current state
        const targetId = 'view-m' + mode + '-s' + step;
        const targetImg = document.getElementById(targetId);

        // 4. Show ONLY the target
        if (targetImg) {
            targetImg.classList.add('visible');
        }

        // 5. Update Button Highlights
        const allBtns = document.querySelectorAll('.mode-btn');
        allBtns.forEach((btn, idx) => {
            if (idx === mode) btn.classList.add('active');
            else btn.classList.remove('active');
        });
    }

    // --- Action: Switch Mode ---
    function selectMode(mode) {
        refreshView(mode, -1);
    }

    // --- Action: Slider Change ---
    function onSliderChange(val) {
        refreshView(-1, parseInt(val));
    }
</script>
"""


empty_html = f"""
<div class="previewer-container">
    <svg style=" opacity: .5; height: var(--size-5); color: var(--body-text-color);"
    xmlns="http://www.w3.org/2000/svg" width="100%" height="100%" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" class="feather feather-image"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect><circle cx="8.5" cy="8.5" r="1.5"></circle><polyline points="21 15 16 10 5 21"></polyline></svg>
</div>
"""


def image_to_base64(image):
    buffered = io.BytesIO()
    image = image.convert("RGB")
    image.save(buffered, format="jpeg", quality=85)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/jpeg;base64,{img_str}"


class _PROCESS_MEMORY_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def _get_rss_mb() -> Optional[float]:
    # Try psutil if available, else fall back to Win32 API.
    try:
        import psutil  # type: ignore
        rss = psutil.Process(os.getpid()).memory_info().rss
        return rss / (1024 * 1024)
    except Exception:
        pass
    if os.name != "nt":
        return None
    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        GetCurrentProcess = kernel32.GetCurrentProcess
        GetCurrentProcess.restype = wintypes.HANDLE
        GetProcessMemoryInfo = psapi.GetProcessMemoryInfo
        GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(_PROCESS_MEMORY_COUNTERS), wintypes.DWORD]
        GetProcessMemoryInfo.restype = wintypes.BOOL
        counters = _PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(_PROCESS_MEMORY_COUNTERS)
        if not GetProcessMemoryInfo(GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            return None
        return counters.WorkingSetSize / (1024 * 1024)
    except Exception:
        return None
    return None


def _dir_snapshot(path: str, limit: int = 20) -> str:
    try:
        items = sorted(os.listdir(path))
    except Exception:
        return "<cannot list>"
    if not items:
        return "<empty>"
    suffix = " ..." if len(items) > limit else ""
    return ", ".join(items[:limit]) + suffix


def _log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def _log_mem(tag: str) -> None:
    rss = _get_rss_mb()
    if rss is None:
        _log(f"[MEM] {tag}: rss=unknown")
    else:
        _log(f"[MEM] {tag}: rss={rss:.1f} MB")


def _log_sys(tag: str) -> None:
    # CPU/RAM via psutil if available.
    cpu_pct = None
    ram_pct = None
    try:
        import psutil  # type: ignore
        cpu_pct = psutil.cpu_percent(interval=0.1)
        ram_pct = psutil.virtual_memory().percent
    except Exception:
        pass
    if cpu_pct is None or ram_pct is None:
        if os.name == "nt":
            try:
                class _FILETIME(ctypes.Structure):
                    _fields_ = [("dwLowDateTime", wintypes.DWORD), ("dwHighDateTime", wintypes.DWORD)]

                def _filetime_to_int(ft: _FILETIME) -> int:
                    return (ft.dwHighDateTime << 32) | ft.dwLowDateTime

                GetSystemTimes = ctypes.windll.kernel32.GetSystemTimes
                GetSystemTimes.argtypes = [ctypes.POINTER(_FILETIME), ctypes.POINTER(_FILETIME), ctypes.POINTER(_FILETIME)]
                GetSystemTimes.restype = wintypes.BOOL

                idle1 = _FILETIME()
                kernel1 = _FILETIME()
                user1 = _FILETIME()
                idle2 = _FILETIME()
                kernel2 = _FILETIME()
                user2 = _FILETIME()
                if GetSystemTimes(ctypes.byref(idle1), ctypes.byref(kernel1), ctypes.byref(user1)):
                    time.sleep(0.1)
                    if GetSystemTimes(ctypes.byref(idle2), ctypes.byref(kernel2), ctypes.byref(user2)):
                        idle = _filetime_to_int(idle2) - _filetime_to_int(idle1)
                        kernel = _filetime_to_int(kernel2) - _filetime_to_int(kernel1)
                        user = _filetime_to_int(user2) - _filetime_to_int(user1)
                        total = kernel + user
                        if total > 0:
                            cpu_pct = (1.0 - (idle / total)) * 100.0

                class _MEMORYSTATUSEX(ctypes.Structure):
                    _fields_ = [
                        ("dwLength", wintypes.DWORD),
                        ("dwMemoryLoad", wintypes.DWORD),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                    ]

                GlobalMemoryStatusEx = ctypes.windll.kernel32.GlobalMemoryStatusEx
                GlobalMemoryStatusEx.argtypes = [ctypes.POINTER(_MEMORYSTATUSEX)]
                GlobalMemoryStatusEx.restype = wintypes.BOOL
                mem = _MEMORYSTATUSEX()
                mem.dwLength = ctypes.sizeof(_MEMORYSTATUSEX)
                if GlobalMemoryStatusEx(ctypes.byref(mem)):
                    ram_pct = float(mem.dwMemoryLoad)
            except Exception:
                pass
    if cpu_pct is not None and ram_pct is not None:
        _log(f"[SYS] {tag}: cpu={cpu_pct:.1f}% ram={ram_pct:.1f}%")
    else:
        _log(f"[SYS] {tag}: cpu=unknown ram=unknown")

    # GPU via nvidia-smi if available.
    try:
        if shutil.which("nvidia-smi"):
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                check=False,
            )
            out = (result.stdout or "").strip().splitlines()
            if out:
                _log(f"[GPU] {tag}: util={out[0].strip()}")
                return
    except Exception:
        pass
    _log(f"[GPU] {tag}: util=unknown")


def _get_sys_usage():
    cpu_pct = None
    ram_pct = None
    try:
        import psutil  # type: ignore
        cpu_pct = psutil.cpu_percent(interval=0.1)
        ram_pct = psutil.virtual_memory().percent
    except Exception:
        pass
    if cpu_pct is None or ram_pct is None:
        if os.name == "nt":
            try:
                class _FILETIME(ctypes.Structure):
                    _fields_ = [("dwLowDateTime", wintypes.DWORD), ("dwHighDateTime", wintypes.DWORD)]

                def _filetime_to_int(ft: _FILETIME) -> int:
                    return (ft.dwHighDateTime << 32) | ft.dwLowDateTime

                GetSystemTimes = ctypes.windll.kernel32.GetSystemTimes
                GetSystemTimes.argtypes = [ctypes.POINTER(_FILETIME), ctypes.POINTER(_FILETIME), ctypes.POINTER(_FILETIME)]
                GetSystemTimes.restype = wintypes.BOOL

                idle1 = _FILETIME()
                kernel1 = _FILETIME()
                user1 = _FILETIME()
                idle2 = _FILETIME()
                kernel2 = _FILETIME()
                user2 = _FILETIME()
                if GetSystemTimes(ctypes.byref(idle1), ctypes.byref(kernel1), ctypes.byref(user1)):
                    time.sleep(0.1)
                    if GetSystemTimes(ctypes.byref(idle2), ctypes.byref(kernel2), ctypes.byref(user2)):
                        idle = _filetime_to_int(idle2) - _filetime_to_int(idle1)
                        kernel = _filetime_to_int(kernel2) - _filetime_to_int(kernel1)
                        user = _filetime_to_int(user2) - _filetime_to_int(user1)
                        total = kernel + user
                        if total > 0:
                            cpu_pct = (1.0 - (idle / total)) * 100.0

                class _MEMORYSTATUSEX(ctypes.Structure):
                    _fields_ = [
                        ("dwLength", wintypes.DWORD),
                        ("dwMemoryLoad", wintypes.DWORD),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                    ]

                GlobalMemoryStatusEx = ctypes.windll.kernel32.GlobalMemoryStatusEx
                GlobalMemoryStatusEx.argtypes = [ctypes.POINTER(_MEMORYSTATUSEX)]
                GlobalMemoryStatusEx.restype = wintypes.BOOL
                mem = _MEMORYSTATUSEX()
                mem.dwLength = ctypes.sizeof(_MEMORYSTATUSEX)
                if GlobalMemoryStatusEx(ctypes.byref(mem)):
                    ram_pct = float(mem.dwMemoryLoad)
            except Exception:
                pass
    return cpu_pct, ram_pct


def _get_gpu_usage():
    try:
        if shutil.which("nvidia-smi"):
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                check=False,
            )
            out = (result.stdout or "").strip().splitlines()
            if out:
                parts = [p.strip() for p in out[0].split(",")]
                if len(parts) >= 3:
                    util = float(parts[0])
                    mem_used = float(parts[1])
                    mem_total = float(parts[2])
                    return util, mem_used, mem_total
    except Exception:
        pass
    return None, None, None

class _TeeStdout:
    def __init__(self, log_path: str):
        self._log_path = log_path
        self._log_f = open(log_path, "a", encoding="utf-8", buffering=1)
        self._stdout = sys.__stdout__

    def write(self, data):
        self._stdout.write(data)
        self._log_f.write(data)

    def flush(self):
        self._stdout.flush()
        self._log_f.flush()

    def isatty(self):
        return True

    def close(self):
        try:
            self._log_f.close()
        except Exception:
            pass


def _stable_baseline_rss(user_dir: str, timeout_sec: int = 60, interval_sec: int = 5, stable_delta_mb: float = 300.0) -> float:
    """
    Try to find a stable RSS baseline. If not stable within timeout, fall back
    to median, then last baseline, then min+1GB.
    """
    samples = []
    start = time.time()
    while time.time() - start < timeout_sec:
        rss = _get_rss_mb()
        if rss is not None:
            samples.append(float(rss))
        if len(samples) >= 2:
            if max(samples) - min(samples) <= stable_delta_mb:
                return sum(samples) / len(samples)
        time.sleep(interval_sec)

    # Fallbacks
    if samples:
        sorted_samples = sorted(samples)
        median = sorted_samples[len(sorted_samples) // 2]
    else:
        median = None

    baseline_path = os.path.join(user_dir, "rss_baseline.json")
    prev = None
    try:
        with open(baseline_path, "r", encoding="utf-8") as f:
            prev = json.load(f).get("baseline_mb")
    except Exception:
        pass

    if median is not None:
        return median
    if prev is not None:
        return float(prev)
    # last resort: 0 baseline
    return 0.0


def _save_baseline_rss(user_dir: str, baseline_mb: float) -> None:
    try:
        with open(os.path.join(user_dir, "rss_baseline.json"), "w", encoding="utf-8") as f:
            json.dump({"baseline_mb": float(baseline_mb)}, f)
    except Exception:
        pass


def _get_user_dir(req: gr.Request) -> str:
    session = str(req.session_hash)
    with _SESSION_DIRS_LOCK:
        if session not in _SESSION_DIRS:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            _SESSION_DIRS[session] = os.path.join(TMP_DIR, f"{ts}_{session}")
        return _SESSION_DIRS[session]


# --- Extract worker pool (simple persistent worker) ---
_extract_pool = {"proc": None, "stdin": None, "stdout": None, "lock": threading.Lock()}


def _ensure_extract_worker(env) -> None:
    with _extract_pool["lock"]:
        proc = _extract_pool["proc"]
        if proc is not None and proc.poll() is None:
            return
        py = sys.executable
        worker = os.path.join(os.path.dirname(os.path.abspath(__file__)), "extract_worker_pool.py")
        proc = subprocess.Popen(
            [py, worker],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            creationflags=0x08000000,
        )
        _extract_pool["proc"] = proc
        _extract_pool["stdin"] = proc.stdin
        _extract_pool["stdout"] = proc.stdout


def _run_extract_via_pool(payload: dict, progress_path: str, worker_log: str, timeout_sec: int = 1800):
    env = {**os.environ, "TQDM_DISABLE": "1"}
    _ensure_extract_worker(env)
    with _extract_pool["lock"]:
        proc = _extract_pool["proc"]
        stdin = _extract_pool["stdin"]
        stdout = _extract_pool["stdout"]
        if proc is None or stdin is None or stdout is None or proc.poll() is not None:
            raise RuntimeError("Extract pool not available")

        # write request
        stdin.write(json.dumps(payload) + "\n")
        stdin.flush()

        # stats sampler
        stop_evt = threading.Event()
        stats = {
            "max_rss": None,
            "max_cpu": None,
            "max_ram": None,
            "max_gpu": None,
            "max_vram": None,
            "vram_total": None,
            "cpu_samples": [],
            "ram_samples": [],
            "gpu_samples": [],
            "vram_samples": [],
        }

        def _sample_loop():
            while not stop_evt.is_set():
                rss = _get_rss_mb()
                if rss is not None:
                    stats["max_rss"] = rss if stats["max_rss"] is None else max(stats["max_rss"], rss)
                cpu_pct, ram_pct = _get_sys_usage()
                if cpu_pct is not None:
                    stats["max_cpu"] = cpu_pct if stats["max_cpu"] is None else max(stats["max_cpu"], cpu_pct)
                    stats["cpu_samples"].append(float(cpu_pct))
                if ram_pct is not None:
                    stats["max_ram"] = ram_pct if stats["max_ram"] is None else max(stats["max_ram"], ram_pct)
                    stats["ram_samples"].append(float(ram_pct))
                gpu_util, mem_used, mem_total = _get_gpu_usage()
                if gpu_util is not None:
                    stats["max_gpu"] = gpu_util if stats["max_gpu"] is None else max(stats["max_gpu"], gpu_util)
                    stats["gpu_samples"].append(float(gpu_util))
                if mem_used is not None:
                    stats["max_vram"] = mem_used if stats["max_vram"] is None else max(stats["max_vram"], mem_used)
                    stats["vram_samples"].append(float(mem_used))
                if mem_total is not None:
                    stats["vram_total"] = mem_total
                time.sleep(1)

        t = threading.Thread(target=_sample_loop, daemon=True)
        t.start()

        # read lines until END marker
        lines = []
        start = time.time()
        while True:
            if timeout_sec is not None and (time.time() - start) > timeout_sec:
                stop_evt.set()
                raise RuntimeError("Extract pool timed out")
            line = stdout.readline()
            if not line:
                stop_evt.set()
                raise RuntimeError("Extract pool terminated unexpectedly")
            lines.append(line)
            if line.strip() == "__END__":
                break

        stop_evt.set()

        # save worker log
        with open(worker_log, "w", encoding="utf-8") as f:
            f.writelines(lines)

        # save stats
        stats_path = payload.get("stats_path", "")
        if stats_path:
            try:
                with open(stats_path, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "max_rss_mb": stats["max_rss"],
                            "max_cpu_pct": stats["max_cpu"],
                            "max_ram_pct": stats["max_ram"],
                            "max_gpu_util_pct": stats["max_gpu"],
                            "max_vram_used_mb": stats["max_vram"],
                            "vram_total_mb": stats["vram_total"],
                            "median_cpu_pct": _pct(stats["cpu_samples"], 0.5),
                            "p90_cpu_pct": _pct(stats["cpu_samples"], 0.9),
                            "median_ram_pct": _pct(stats["ram_samples"], 0.5),
                            "p90_ram_pct": _pct(stats["ram_samples"], 0.9),
                            "median_gpu_util_pct": _pct(stats["gpu_samples"], 0.5),
                            "p90_gpu_util_pct": _pct(stats["gpu_samples"], 0.9),
                            "median_vram_used_mb": _pct(stats["vram_samples"], 0.5),
                            "p90_vram_used_mb": _pct(stats["vram_samples"], 0.9),
                        },
                        f,
                    )
            except Exception:
                pass

        # find glb path from log lines
        glb_path = ""
        for line in reversed(lines):
            if line.strip().endswith(".glb"):
                glb_path = line.strip()
                break

        return 0, glb_path


def _tail_file(path: str, max_lines: int = 50) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        return "".join(lines[-max_lines:]).strip()
    except Exception:
        return "<cannot read log>"


def _read_progress(progress_path: Optional[str]):
    if not progress_path:
        return None
    try:
        with open(progress_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _pct(samples, q):
    if not samples:
        return None
    s = sorted(samples)
    idx = int((len(s) - 1) * q)
    return s[idx]


def _run_worker_with_progress(args, cwd, log_path, env, label, timeout_sec, progress_path=None, stats_path=None):
    start = time.time()
    max_rss = None
    max_cpu = None
    max_ram = None
    max_gpu = None
    max_vram = None
    vram_total = None
    cpu_samples = []
    ram_samples = []
    gpu_samples = []
    vram_samples = []
    with open(log_path, "w", encoding="utf-8") as logf:
        proc = subprocess.Popen(
            args,
            cwd=cwd,
            stdout=logf,
            stderr=subprocess.STDOUT,
            text=True,
            stdin=subprocess.DEVNULL,
            env=env,
            creationflags=0x08000000,
        )
        bar_width = 30
        last_line = ""
        last_len = 0
        while True:
            ret = proc.poll()
            elapsed = int(time.time() - start)
            progress = _read_progress(progress_path)
            if progress and isinstance(progress, dict):
                pct = int(progress.get("pct", 0))
                stage = str(progress.get("stage", "running"))
                pct = max(0, min(100, pct))
                filled = int((pct / 100) * bar_width)
                bar = "#" * filled + "-" * (bar_width - filled)
                line = f"{label}: [{bar}] {pct:3d}% {elapsed}s ({stage})"
            elif timeout_sec:
                ratio = min(elapsed / timeout_sec, 1.0)
                filled = int(ratio * bar_width)
                bar = "#" * filled + "-" * (bar_width - filled)
                pct = int(ratio * 100)
                line = f"{label}: [{bar}] {pct:3d}% {elapsed}s/{timeout_sec}s"
            else:
                spinner = "|/-\\"[elapsed % 4]
                line = f"{label}: {spinner} {elapsed}s"

            if line != last_line:
                if len(line) < last_len:
                    line = line + (" " * (last_len - len(line)))
                sys.stdout.write("\r" + line)
                sys.stdout.flush()
                last_line = line
                last_len = len(line)

            if ret is not None:
                sys.stdout.write("\r" + line + " done\n")
                sys.stdout.flush()
                if stats_path:
                    try:
                        with open(stats_path, "w", encoding="utf-8") as f:
                            json.dump(
                                {
                                    "max_rss_mb": max_rss,
                                    "max_cpu_pct": max_cpu,
                                    "max_ram_pct": max_ram,
                                    "max_gpu_util_pct": max_gpu,
                                    "max_vram_used_mb": max_vram,
                                    "vram_total_mb": vram_total,
                                    "median_cpu_pct": _pct(cpu_samples, 0.5),
                                    "p90_cpu_pct": _pct(cpu_samples, 0.9),
                                    "median_ram_pct": _pct(ram_samples, 0.5),
                                    "p90_ram_pct": _pct(ram_samples, 0.9),
                                    "median_gpu_util_pct": _pct(gpu_samples, 0.5),
                                    "p90_gpu_util_pct": _pct(gpu_samples, 0.9),
                                    "median_vram_used_mb": _pct(vram_samples, 0.5),
                                    "p90_vram_used_mb": _pct(vram_samples, 0.9),
                                },
                                f,
                            )
                    except Exception:
                        pass
                return ret

            rss = _get_rss_mb()
            if rss is not None:
                max_rss = rss if max_rss is None else max(max_rss, rss)
            cpu_pct, ram_pct = _get_sys_usage()
            if cpu_pct is not None:
                max_cpu = cpu_pct if max_cpu is None else max(max_cpu, cpu_pct)
                cpu_samples.append(float(cpu_pct))
            if ram_pct is not None:
                max_ram = ram_pct if max_ram is None else max(max_ram, ram_pct)
                ram_samples.append(float(ram_pct))
            gpu_util, mem_used, mem_total = _get_gpu_usage()
            if gpu_util is not None:
                max_gpu = gpu_util if max_gpu is None else max(max_gpu, gpu_util)
                gpu_samples.append(float(gpu_util))
            if mem_used is not None:
                max_vram = mem_used if max_vram is None else max(max_vram, mem_used)
                vram_samples.append(float(mem_used))
            if mem_total is not None:
                vram_total = mem_total

            time.sleep(1)
            if timeout_sec is not None and elapsed > timeout_sec:
                proc.kill()
                raise subprocess.TimeoutExpired(args, timeout_sec)


def start_session(req: gr.Request):
    user_dir = _get_user_dir(req)
    os.makedirs(user_dir, exist_ok=True)
    _log(f"[EXTRACT] start session={req.session_hash} dir={user_dir} files={_dir_snapshot(user_dir)}")
    _log_mem("extract_glb:start")
    _log_sys("extract_glb:start")
    _log(f"[SESSION] start {req.session_hash} dir={user_dir} files={_dir_snapshot(user_dir)}")
    _log_mem("start_session")
    _log_sys("start_session")


def end_session(req: gr.Request):
    user_dir = _get_user_dir(req)
    shutil.rmtree(user_dir, ignore_errors=True)
    _log(f"[SESSION] end {req.session_hash} dir={user_dir} removed")
    _log_mem("end_session")
    _log_sys("end_session")


def preprocess_image(image: Image.Image) -> Image.Image:
    processed_image = pipeline.preprocess_image(image)
    return processed_image


# --- Memory-friendly state: save tensors to disk instead of embedding big numpy arrays into gr.State ---
def pack_state_to_disk(latents: Tuple[SparseTensor, SparseTensor, int], user_dir: str) -> dict:
    shape_slat, tex_slat, res = latents
    os.makedirs(user_dir, exist_ok=True)
    state_path = os.path.join(user_dir, "latent_state.pt")
    torch.save({
        "shape_feats": shape_slat.feats.detach().cpu(),
        "tex_feats": tex_slat.feats.detach().cpu(),
        "coords": shape_slat.coords.detach().cpu(),
        "res": int(res),
    }, state_path)
    return {"state_path": state_path}


def unpack_state_from_disk(state: dict) -> Tuple[SparseTensor, SparseTensor, int]:
    data = torch.load(state["state_path"], map_location="cpu")
    shape_slat = SparseTensor(
        feats=data["shape_feats"].cuda(),
        coords=data["coords"].cuda(),
    )
    tex_slat = shape_slat.replace(data["tex_feats"].cuda())
    return shape_slat, tex_slat, int(data["res"])


def get_seed(randomize_seed: bool, seed: int) -> int:
    return np.random.randint(0, MAX_SEED) if randomize_seed else seed


def image_to_3d(
    image: Image.Image,
    seed: int,
    resolution: str,
    ss_guidance_strength: float,
    ss_guidance_rescale: float,
    ss_sampling_steps: int,
    ss_rescale_t: float,
    shape_slat_guidance_strength: float,
    shape_slat_guidance_rescale: float,
    shape_slat_sampling_steps: int,
    shape_slat_rescale_t: float,
    tex_slat_guidance_strength: float,
    tex_slat_guidance_rescale: float,
    tex_slat_sampling_steps: int,
    tex_slat_rescale_t: float,
    req: gr.Request,
    progress=gr.Progress(track_tqdm=True),
):
    user_dir = _get_user_dir(req)
    _log(f"[GEN] start session={req.session_hash} dir={user_dir} files={_dir_snapshot(user_dir)}")
    _log_mem("image_to_3d:start")
    _log_sys("image_to_3d:start")
    baseline = _stable_baseline_rss(user_dir)
    _save_baseline_rss(user_dir, baseline)
    _log(f"[MEM] baseline_rss={baseline:.1f} MB")
    # --- Sampling in worker process ---
    os.makedirs(user_dir, exist_ok=True)
    lock_path = os.path.join(user_dir, "generate.lock")
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_RDWR)
        os.close(fd)
    except FileExistsError:
        raise RuntimeError("Generate already running for this session. Please wait for it to finish.")

    worker_log = os.path.join(user_dir, f"gen_worker_{time.strftime('%Y%m%d_%H%M%S')}.log")
    progress_path = os.path.join(user_dir, f"gen_worker_progress_{time.strftime('%Y%m%d_%H%M%S')}.json")
    stats_path = os.path.join(user_dir, "gen_worker_stats.json")
    with io.BytesIO() as buf:
        image.save(buf, format="PNG")
        image_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    payload = {
        "image_b64": image_b64,
        "seed": seed,
        "resolution": resolution,
        "out_dir": user_dir,
        "preview_resolution": PREVIEW_RESOLUTION,
        "preview_views": NORMAL_PREVIEW_VIEWS,
        "ss": {
            "steps": ss_sampling_steps,
            "guidance_strength": ss_guidance_strength,
            "guidance_rescale": ss_guidance_rescale,
            "rescale_t": ss_rescale_t,
        },
        "shape": {
            "steps": shape_slat_sampling_steps,
            "guidance_strength": shape_slat_guidance_strength,
            "guidance_rescale": shape_slat_guidance_rescale,
            "rescale_t": shape_slat_rescale_t,
        },
        "tex": {
            "steps": tex_slat_sampling_steps,
            "guidance_strength": tex_slat_guidance_strength,
            "guidance_rescale": tex_slat_guidance_rescale,
            "rescale_t": tex_slat_rescale_t,
        },
        "progress_path": progress_path,
    }

    py = sys.executable
    worker = os.path.join(os.path.dirname(os.path.abspath(__file__)), "generate_worker_patched.py")
    payload_path = os.path.join(user_dir, "gen_worker_payload.json")
    with open(payload_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    try:
        _log(f"[GEN] worker spawn {worker} log={worker_log}")
        ret = _run_worker_with_progress(
            [py, worker, payload_path],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            log_path=worker_log,
            env={**os.environ, "TQDM_DISABLE": "1"},
            label="Generate worker",
            progress_path=progress_path,
            stats_path=stats_path,
            timeout_sec=1800,
        )
        _log(f"[GEN] worker exit code={ret} log={worker_log}")
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"Generate worker timed out. See log: {worker_log}")
    finally:
        try:
            os.remove(lock_path)
        except Exception:
            pass

    if ret != 0:
        tail = _tail_file(worker_log, max_lines=80)
        raise RuntimeError(f"Generate worker failed. See log: {worker_log}\n{tail}")

    out_text = _tail_file(worker_log, max_lines=200)
    try:
        result = json.loads(out_text.splitlines()[-1])
    except Exception:
        raise RuntimeError("Generate worker returned invalid output:\n" + out_text)

    state = {"state_path": result.get("state_path", "")}
    if not state["state_path"] or not os.path.exists(state["state_path"]):
        raise RuntimeError("State file not found after worker run.")
    try:
        state_size = os.path.getsize(state["state_path"])
    except Exception:
        state_size = -1
    _log(f"[GEN] state saved {state['state_path']} size={state_size}")

    # --- Gallery Construction (memory-friendly) ---
    gallery = []
    preview_paths = result.get("preview_paths") or []
    for idx, path in enumerate(preview_paths):
        try:
            img = Image.open(path)
            caption = f"Normal | view {idx}"
            gallery.append((img, caption))
        except Exception:
            pass

    torch.cuda.empty_cache()
    gc.collect()
    _log_mem("image_to_3d:after_cleanup")
    _log_sys("image_to_3d:after_cleanup")

    return state, gallery


def render_full_preview(
    state: dict,
    progress=gr.Progress(track_tqdm=True),
):
    shape_slat, tex_slat, res = unpack_state_from_disk(state)
    with torch.no_grad():
        mesh = pipeline.decode_latent(shape_slat, tex_slat, res)[0]
        mesh.simplify(16777216)  # nvdiffrast limit
        images = render_utils.render_snapshot(
            mesh,
            resolution=PREVIEW_RESOLUTION,
            r=2,
            fov=36,
            nviews=FULL_PREVIEW_VIEWS,
            envmap=envmap,
        )

    gallery = []
    for m_idx, mode in enumerate(MODES):
        for s_idx in range(FULL_PREVIEW_VIEWS):
            img = Image.fromarray(images[mode["render_key"]][s_idx])
            caption = f"{mode['name']} | view {s_idx}"
            gallery.append((img, caption))

    # Cleanup
    del images, mesh, shape_slat, tex_slat
    torch.cuda.empty_cache()
    gc.collect()

    return gallery


def extract_glb(
    state: dict,
    decimation_target: int,
    texture_size: int,
    req: gr.Request,
    progress=gr.Progress(track_tqdm=True),
) -> Tuple[str, str]:
    # Run extraction in a separate process to avoid CUDA extension state carry-over
    import subprocess, json as _json

    user_dir = _get_user_dir(req)
    os.makedirs(user_dir, exist_ok=True)
    worker_log = os.path.join(user_dir, f"extract_worker_{time.strftime('%Y%m%d_%H%M%S')}.log")
    progress_path = os.path.join(user_dir, f"extract_worker_progress_{time.strftime('%Y%m%d_%H%M%S')}.json")
    stats_path = os.path.join(user_dir, "extract_worker_stats.json")
    _log(f"[EXTRACT] worker log={worker_log}")

    state_path = state.get("state_path", "")
    if not state_path or not os.path.exists(state_path):
        raise RuntimeError("State file not found. Please Generate first.")
    try:
        st = os.stat(state_path)
        _log(f"[EXTRACT] state={state_path} size={st.st_size} mtime={int(st.st_mtime)}")
    except Exception:
        _log(f"[EXTRACT] state={state_path} size=unknown mtime=unknown")

    payload = {
        "state_path": state_path,
        "out_dir": user_dir,
        "decimation_target": int(decimation_target),
        "texture_size": int(texture_size),
        "remesh": True,  # heavier, but can fix invalid normals during GLB extraction
        "progress_path": progress_path,
        "stats_path": stats_path,
    }

    py = sys.executable
    worker = os.path.join(os.path.dirname(os.path.abspath(__file__)), "extract_worker_patched.py")

    ret, glb_path = _run_extract_via_pool(
        payload=payload,
        progress_path=progress_path,
        worker_log=worker_log,
        timeout_sec=1800,
    )
    _log(f"[EXTRACT] worker exit code={ret} log={worker_log}")
    # If worker failed, surface BOTH stderr and stdout (tqdm often writes to stderr).
    if ret != 0:
        tail = _tail_file(worker_log, max_lines=120)
        raise RuntimeError("Extract worker failed:\n" + tail)

    if not glb_path:
        out_text = _tail_file(worker_log, max_lines=200)
        glb_path = out_text.splitlines()[-1].strip() if out_text else ""

    # Fallback: if stdout is empty or file missing, pick newest .glb in user_dir.
    if (not glb_path) or (not os.path.exists(glb_path)):
        import glob as _glob, os as _os
        candidates = sorted(_glob.glob(_os.path.join(user_dir, "*.glb")), key=_os.path.getmtime, reverse=True)
        if candidates:
            glb_path = candidates[0]
        else:
            dbg = ("[stderr]\n" + (proc.stderr or "") + "\n[stdout]\n" + (proc.stdout or "")).strip()
            try:
                listing = "\n".join(sorted(_os.listdir(user_dir))[:50])
            except Exception:
                listing = "<cannot list out_dir>"
            raise RuntimeError("Worker finished but GLB not found.\n" + dbg + "\n[out_dir listing]\n" + listing)

    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()
    _log_mem("extract_glb:after_cleanup")
    _log_sys("extract_glb:after_cleanup")

    return glb_path, glb_path



def purge_session(req: gr.Request):
    """Clear per-session temp files and encourage Python/PyTorch to release memory."""
    user_dir = _get_user_dir(req)
    _log(f"[PURGE] start session={req.session_hash} dir={user_dir} files={_dir_snapshot(user_dir)}")
    _log_mem("purge_session:before")
    _log_sys("purge_session:before")
    # Remove cached files but keep logs
    if os.path.isdir(user_dir):
        for name in os.listdir(user_dir):
            path = os.path.join(user_dir, name)
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
                continue
            # Keep log files
            if name.lower().endswith(".log"):
                continue
            try:
                os.remove(path)
            except Exception:
                pass
    else:
        os.makedirs(user_dir, exist_ok=True)

    # Aggressive cleanup
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()
    _log(f"[PURGE] done session={req.session_hash} dir={user_dir} files={_dir_snapshot(user_dir)}")
    _log_mem("purge_session:after")
    _log_sys("purge_session:after")

    # Clear UI state/outputs
    empty_gallery = []
    # For Model3D + DownloadButton we can return None to clear
    return None, empty_gallery, None, None, gr.Walkthrough(selected=0)

with gr.Blocks(delete_cache=(600, 600)) as demo:
    gr.Markdown("""
    ## Image to 3D Asset with [TRELLIS.2](https://microsoft.github.io/TRELLIS.2)
    * Upload an image (preferably with an alpha-masked foreground object) and click Generate to create a 3D asset.
    * Click Extract GLB to export and download the generated GLB file if you're satisfied with the result. Otherwise, try another time.
    """)

    with gr.Row():
        with gr.Column(scale=1, min_width=360):
            image_prompt = gr.Image(label="Image Prompt", format="png", image_mode="RGBA", type="pil", height=400)

            # Default to 512 to avoid OOM (was 1024)
            resolution = gr.Radio(["512", "1024", "1536"], label="Resolution", value="512")
            seed = gr.Slider(0, MAX_SEED, label="Seed", value=1234, step=1)
            randomize_seed = gr.Checkbox(label="Randomize Seed", value=False)

            # Lower defaults to reduce CuMesh OOM
            decimation_target = gr.Slider(100000, 1000000, label="Decimation Target", value=200000, step=10000)
            texture_size = gr.Slider(512, 4096, label="Texture Size", value=1024, step=512)

            generate_btn = gr.Button("Generate")

            with gr.Accordion(label="Advanced Settings", open=False):
                gr.Markdown("Stage 1: Sparse Structure Generation")
                with gr.Row():
                    ss_guidance_strength = gr.Slider(1.0, 10.0, label="Guidance Strength", value=7.5, step=0.1)
                    ss_guidance_rescale = gr.Slider(0.0, 1.0, label="Guidance Rescale", value=0.7, step=0.01)
                    ss_sampling_steps = gr.Slider(1, 50, label="Sampling Steps", value=12, step=1)
                    ss_rescale_t = gr.Slider(1.0, 6.0, label="Rescale T", value=5.0, step=0.1)
                gr.Markdown("Stage 2: Shape Generation")
                with gr.Row():
                    shape_slat_guidance_strength = gr.Slider(1.0, 10.0, label="Guidance Strength", value=7.5, step=0.1)
                    shape_slat_guidance_rescale = gr.Slider(0.0, 1.0, label="Guidance Rescale", value=0.5, step=0.01)
                    shape_slat_sampling_steps = gr.Slider(1, 50, label="Sampling Steps", value=12, step=1)
                    shape_slat_rescale_t = gr.Slider(1.0, 6.0, label="Rescale T", value=3.0, step=0.1)
                gr.Markdown("Stage 3: Material Generation")
                with gr.Row():
                    tex_slat_guidance_strength = gr.Slider(1.0, 10.0, label="Guidance Strength", value=1.0, step=0.1)
                    tex_slat_guidance_rescale = gr.Slider(0.0, 1.0, label="Guidance Rescale", value=0.0, step=0.01)
                    tex_slat_sampling_steps = gr.Slider(1, 50, label="Sampling Steps", value=12, step=1)
                    tex_slat_rescale_t = gr.Slider(1.0, 6.0, label="Rescale T", value=3.0, step=0.1)

        with gr.Column(scale=10):
            with gr.Walkthrough(selected=0) as walkthrough:
                with gr.Step("Preview", id=0):
                    preview_output = gr.Gallery(label="3D Asset Preview", show_label=True, columns=4, height=724, object_fit="contain")
                    with gr.Row():
                        extract_btn = gr.Button("Extract GLB")
                        full_preview_btn = gr.Button("Generate All Modes")
                        purge_btn = gr.Button("Purge / Clear Memory")
                with gr.Step("Extract", id=1):
                    glb_output = gr.Model3D(label="Extracted GLB", height=724, show_label=True, display_mode="solid", clear_color=(0.25, 0.25, 0.25, 1.0))
                    download_btn = gr.DownloadButton(label="Download GLB")

        with gr.Column(scale=1, min_width=172):
            examples = gr.Examples(
                examples=[
                    f'assets/example_image/{image}'
                    for image in os.listdir("assets/example_image")
                ],
                inputs=[image_prompt],
                fn=preprocess_image,
                outputs=[image_prompt],
                run_on_click=True,
                examples_per_page=18,
            )

    output_buf = gr.State()

    # Handlers
    demo.load(start_session)
    demo.unload(end_session)

    image_prompt.upload(
        preprocess_image,
        inputs=[image_prompt],
        outputs=[image_prompt],
        queue=False,
    )

    generate_btn.click(
        get_seed,
        inputs=[randomize_seed, seed],
        outputs=[seed],
        queue=True,
    ).then(
        lambda: gr.Walkthrough(selected=0), outputs=walkthrough,
        queue=True,
    ).then(
        image_to_3d,
        inputs=[
            image_prompt, seed, resolution,
            ss_guidance_strength, ss_guidance_rescale, ss_sampling_steps, ss_rescale_t,
            shape_slat_guidance_strength, shape_slat_guidance_rescale, shape_slat_sampling_steps, shape_slat_rescale_t,
            tex_slat_guidance_strength, tex_slat_guidance_rescale, tex_slat_sampling_steps, tex_slat_rescale_t,
        ],
        outputs=[output_buf, preview_output],
        queue=True,
    )

    extract_btn.click(
        lambda: gr.Walkthrough(selected=1), outputs=walkthrough,
        queue=False,
    ).then(
        extract_glb,
        inputs=[output_buf, decimation_target, texture_size],
        outputs=[glb_output, download_btn],
        queue=False,
    )
    full_preview_btn.click(
        render_full_preview,
        inputs=[output_buf],
        outputs=[preview_output],
        queue=False,
    )
    purge_btn.click(
        purge_session,
        inputs=[],
        outputs=[output_buf, preview_output, glb_output, download_btn, walkthrough],
        queue=False,
    )



# Launch the Gradio app
if __name__ == "__main__":
    # Mirror stdout to a per-run log while keeping TTY behavior (progress bars)
    run_log = Path(TMP_DIR) / f"app_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    sys.stdout = _TeeStdout(str(run_log))
    sys.stderr = sys.stdout
    print(f"[BOOT] log={run_log}", flush=True)
    os.makedirs(TMP_DIR, exist_ok=True)

    # Construct ui components
    for i in range(len(MODES)):
        icon = Image.open(MODES[i]['icon'])
        MODES[i]['icon_base64'] = image_to_base64(icon)

    pipeline = Trellis2ImageTo3DPipeline.from_pretrained('microsoft/TRELLIS.2-4B')
    pipeline.cuda()

    envmap = {
        'forest': EnvMap(torch.tensor(
            cv2.cvtColor(cv2.imread('assets/hdri/forest.exr', cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB),
            dtype=torch.float32, device='cuda'
        )),
        'sunset': EnvMap(torch.tensor(
            cv2.cvtColor(cv2.imread('assets/hdri/sunset.exr', cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB),
            dtype=torch.float32, device='cuda'
        )),
        'courtyard': EnvMap(torch.tensor(
            cv2.cvtColor(cv2.imread('assets/hdri/courtyard.exr', cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB),
            dtype=torch.float32, device='cuda'
        )),
    }

    demo.launch(css=css)
