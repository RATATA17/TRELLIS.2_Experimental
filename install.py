# install.py
# File: install.py
import subprocess
import sys
import os
import time
from typing import Optional, Tuple
from pathlib import Path
import urllib.request
import urllib.error
import socket
import shutil
import traceback
import importlib

MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds
REQUIRED_PYTHON = (3, 11) # Matches cp311 wheels

class InstallationError(Exception):
    """Custom exception for installation failures"""
    pass


def patch_flex_gemm_triton_import() -> bool:
    """Patch flex_gemm kernels init to tolerate missing triton submodule on Windows wheels."""
    try:
        site_pkgs = Path(next(p for p in sys.path if p.endswith("site-packages")))
    except StopIteration:
        print("[WARN] site-packages path not found; skip flex_gemm patch.")
        return False

    target = site_pkgs / "flex_gemm" / "kernels" / "__init__.py"
    if not target.exists():
        print("[INFO] flex_gemm kernels init not found; skip patch.")
        return False

    text = target.read_text(encoding="utf-8", errors="ignore")
    if "from . import triton" not in text:
        print("[INFO] flex_gemm triton import pattern not found; skip patch.")
        return False

    if "class _TritonFallback" in text or "try:" in text and "from . import triton" in text:
        print("[INFO] flex_gemm triton fallback already present.")
        return False

    patched = text.replace(
        "from . import triton\nfrom . import cuda\n",
        "try:\n    from . import triton\nexcept Exception:\n    class _TritonFallback:\n        @staticmethod\n        def indice_weighed_sum_fwd(feats, indices, weights):\n            import torch\n            N = feats.shape[0]\n            idx = indices.long().clamp(min=0, max=N - 1)\n            M_shape, K = idx.shape\n            C = feats.shape[-1]\n            out = torch.zeros((M_shape, C), dtype=feats.dtype, device=feats.device)\n            for i in range(K):\n                out += feats[idx[:, i]] * weights[:, i].unsqueeze(-1)\n            return out\n\n        @staticmethod\n        def indice_weighed_sum_bwd_input(grad_output, indices, weights, N):\n            import torch\n            M, C = grad_output.shape\n            idx = indices.long().clamp(min=0, max=N - 1)\n            weighted_grad = grad_output.unsqueeze(1) * weights.unsqueeze(-1)\n            grad_feats = torch.zeros(N, C, device=grad_output.device, dtype=grad_output.dtype)\n            grad_feats.scatter_add_(0, idx.reshape(-1, 1).expand(-1, C), weighted_grad.reshape(-1, C))\n            return grad_feats\n\n    triton = _TritonFallback()\nfrom . import cuda\n",
        1,
    )

    if patched == text:
        print("[WARN] flex_gemm patch pattern mismatch; no change applied.")
        return False

    target.write_text(patched, encoding="utf-8")
    print(f"[INFO] Applied flex_gemm triton fallback patch: {target}")
    return True

def get_current_script_dir() -> Path:
    """Helper to get the directory of the current script."""
    try:
        return Path(__file__).parent.resolve()
    except NameError:
        return Path(os.getcwd()).resolve()

def check_python_version():
    """Ensure we are running on the correct Python version for the wheels."""
    current = sys.version_info[:2]
    if current != REQUIRED_PYTHON:
        print(f"Error: This installer requires Python {REQUIRED_PYTHON[0]}.{REQUIRED_PYTHON[1]}")
        print(f"You are currently using Python {current[0]}.{current[1]}")
        print("Please use the embedded python or correct environment.")
        sys.exit(1)

def check_connectivity(url: str = "https://pytorch.org", timeout: int = 5) -> Tuple[bool, Optional[str]]:
    """Check internet connectivity."""
    try:
        urllib.request.urlopen(url, timeout=timeout)
        return True, None
    except urllib.error.URLError as e:
        reason = getattr(e, 'reason', str(e))
        if isinstance(reason, socket.gaierror):
            return False, f"DNS resolution failed: {reason}"
        elif isinstance(reason, socket.timeout) or 'timed out' in str(e):
            return False, "Connection timed out"
        else:
            return False, f"Connection failed: {reason}"
    except Exception as e:
        return False, f"Unknown error: {str(e)}"

def get_git_env() -> dict:
    """Return a copy of the current environment configured to use the portable Git."""
    env = os.environ.copy()
    CODE_DIR = get_current_script_dir()
    PORTABLE_GIT_BASE = (CODE_DIR / ".." / "tools" / "git").resolve()
    
    if PORTABLE_GIT_BASE.exists():
        git_paths = [
            str(PORTABLE_GIT_BASE / "mingw64" / "bin"),
            str(PORTABLE_GIT_BASE / "cmd"),
            str(PORTABLE_GIT_BASE / "usr" / "bin"),
        ]
        existing_path = env.get("PATH", "")
        env["PATH"] = ";".join(git_paths) + (";" + existing_path if existing_path else "")
        
        ca_bundle = PORTABLE_GIT_BASE / "mingw64" / "etc" / "ssl" / "certs" / "ca-bundle.crt"
        if ca_bundle.exists():
            env["GIT_SSL_CAINFO"] = str(ca_bundle)
            env["SSL_CERT_FILE"]  = str(ca_bundle)
    
    return env

def run_command_with_retry(cmd: str, desc: Optional[str] = None, max_retries: int = MAX_RETRIES, fatal: bool = True) -> subprocess.CompletedProcess:
    """Run a command with retry logic."""
    last_error = None
    env = get_git_env()
    
    if cmd.startswith('pip install'):
        args = cmd[11:]
        cmd = f'"{sys.executable}" -m pip install --no-cache-dir --isolated {args}'

    if "pip install" in cmd and "--progress-bar" not in cmd:
        cmd += " --progress-bar=on"

    for attempt in range(max_retries):
        try:
            if attempt > 0:
                print(f"\nRetry attempt {attempt + 1}/{max_retries} for: {desc or cmd}")
                connected, error_msg = check_connectivity()
                if not connected:
                    print(f"Waiting {RETRY_DELAY} seconds before retry...")
                    time.sleep(RETRY_DELAY)
                    continue 
            
            if "pip install" in cmd:
                result = subprocess.run(cmd, shell=True, text=True, stdout=sys.stdout, stderr=subprocess.PIPE, env=env)
            else:
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True, env=env)
            
            if result.returncode == 0:
                return result
            
            last_error = result
            print(f"\nCommand failed (attempt {attempt + 1}/{max_retries}):")
            if hasattr(result, 'stderr') and result.stderr:
                print(f"Error output:\n{result.stderr}")
            
        except Exception as e:
            last_error = e
            print(f"\nException during {desc or cmd} (attempt {attempt + 1}/{max_retries}):")
            print(str(e))
        
        if attempt < max_retries - 1:
            time.sleep(RETRY_DELAY)
    
    if fatal:
        raise InstallationError(f"Command failed after {max_retries} attempts: {last_error}")
    else:
        print(f"Warning: Command '{desc}' failed. Continuing...")
        return last_error

def install_dependencies():
    """Install Trellis 2 dependencies."""
    CODE_DIR = get_current_script_dir()
    check_python_version()

    try:
        connected, error_msg = check_connectivity()
        if not connected:
            print(f"Error: Internet connectivity check failed: {error_msg}")
            sys.exit(1)
        
        # 1. PyTorch 2.8.0 + CUDA 12.8
        print("\n--- Installing PyTorch 2.8.0 (CUDA 12.8) ---")
        torch_cmd = "pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128"
        run_command_with_retry(torch_cmd, "Installing PyTorch")

        # 2. General Dependencies
        print("\n--- Installing General Dependencies ---")
        general_deps = [
            "imageio==2.37.3",
            "imageio-ffmpeg==0.6.0",
            "tqdm==4.67.3",
            "easydict==1.13",
            "opencv-python-headless==4.13.0.92",
            "ninja==1.13.0",
            "trimesh==4.11.5",
            "transformers==4.57.6",
            "gradio==6.0.1",
            "tensorboard==2.20.0",
            "pandas==2.3.3",
            "lpips==0.1.4",
            "zstandard==0.25.0",
            "kornia==0.8.2",
            "timm==0.9.16",
            "huggingface_hub==0.36.2",
            "accelerate==1.13.0",
            "psutil==7.2.2",
            "triton-windows==3.5.0.post21",
        ]
        run_command_with_retry(f"pip install {' '.join(general_deps)}", "Installing pip packages")

        # 3. Handle Pillow Replacement (Standard -> SIMD)
        print("\n--- Configuring Pillow ---")
        # Attempt to install Pillow-SIMD wheel
        whl_dir = CODE_DIR / "whl"
        simd_wheels = list(whl_dir.glob("Pillow_SIMD*.whl"))
        
        simd_success = False
        if simd_wheels:
            print("Uninstalling standard Pillow...")
            subprocess.run(f'"{sys.executable}" -m pip uninstall -y pillow', shell=True, stdout=subprocess.DEVNULL)
            
            print(f"Installing Pillow-SIMD: {simd_wheels[0].name}")
            try:
                run_command_with_retry(f"pip install {simd_wheels[0]}", "Installing Pillow-SIMD")
                
                # --- SAFETY CHECK ---
                # Verify immediately if Pillow-SIMD actually works
                print("Verifying Pillow-SIMD...")
                check_cmd = f'"{sys.executable}" -c "from PIL import Image; print(\'Pillow OK\')"'
                check_result = subprocess.run(check_cmd, shell=True, capture_output=True, text=True)
                
                if check_result.returncode == 0:
                    print("Pillow-SIMD installed and verified successfully.")
                    simd_success = True
                else:
                    print(f"Warning: Pillow-SIMD installed but failed to load. (Error: {check_result.stderr.strip()})")
                    print("Falling back to standard Pillow...")
            except:
                print("Failed to install Pillow-SIMD wheel. Falling back to standard Pillow...")

        if not simd_success:
            # Revert to standard Pillow if SIMD missing or broken
            # We uninstall Pillow-SIMD first just in case
            subprocess.run(f'"{sys.executable}" -m pip uninstall -y Pillow-SIMD', shell=True, stdout=subprocess.DEVNULL)
            run_command_with_retry("pip install pillow", "Installing Standard Pillow")
        
        # 4. Install Local Wheels
        print("\n--- Installing Custom Wheels ---")
        
        # We map descriptions to glob patterns
        wheel_patterns = {
            "Nvdiffrast": "nvdiffrast*.whl",
            "Nvdiffrec": "nvdiffrec*.whl",
            "CuMesh": "cumesh*.whl",
            "FlexGEMM": "flex_gemm*.whl",
            "O-Voxel": "o_voxel*.whl",
            "Flash Attention": "flash_attn*.whl",
            "Utils3D": "utils3d*.whl"
        }

        for desc, pattern in wheel_patterns.items():
            found = list(whl_dir.glob(pattern))
            if found:
                # Install the first match found (with FlashAttention-specific selection)
                selected = found[0]
                if desc == "Flash Attention":
                    # Prefer wheel explicitly matching current environment tags in filename.
                    # Keep torch pinned by avoiding dependency re-resolution.
                    candidates = sorted(
                        found,
                        key=lambda p: (
                            "torch2.8.0" not in p.name.lower(),
                            "cp311" not in p.name.lower(),
                            "win_amd64" not in p.name.lower(),
                            p.name,
                        )
                    )
                    selected = candidates[0]
                    print(f"Using FlashAttention local wheel: {selected.name}")
                    run_command_with_retry(
                        f"pip install --no-deps {selected}",
                        "Installing Flash Attention Wheel (local, no-deps)"
                    )
                else:
                    run_command_with_retry(f"pip install {selected}", f"Installing {desc} Wheel")
            else:
                if desc == "Utils3D":
                    # Fallback to git for utils3d if not compiled
                    print(f"Utils3D wheel not found. Installing via Git...")
                    run_command_with_retry(
                        "pip install git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8", 
                        "Installing Utils3D (Git)"
                    )
                else:
                    print(f"Warning: Wheel for {desc} (pattern: {pattern}) not found in 'whl' folder.")

        # 5. Apply Windows-specific flex_gemm import fallback if needed
        patch_flex_gemm_triton_import()

        print("\nInstallation completed successfully!")

    except InstallationError as e:
        print(f"\nInstallation failed: {str(e)}")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
        sys.exit(1)

def verify_module_import(module_name: str, required: bool = True, hint: Optional[str] = None) -> bool:
    """Import a module and print detailed diagnostics on failure."""
    try:
        importlib.import_module(module_name)
        print(f"[OK] {module_name} detected.")
        return True
    except Exception as e:
        # Auto-recover once for known flex_gemm triton import issue on Windows wheels.
        if module_name == "flex_gemm" and "cannot import name 'triton'" in str(e):
            print("[WARN] Detected flex_gemm triton import issue. Trying auto-patch and retry...")
            if patch_flex_gemm_triton_import():
                try:
                    importlib.invalidate_caches()
                    importlib.import_module(module_name)
                    print(f"[OK] {module_name} detected after auto-patch.")
                    return True
                except Exception as retry_e:
                    e = retry_e

        level = "ERROR" if required else "WARNING"
        print(f"[{level}] Failed to import '{module_name}': {e}")
        if hint:
            print(f"[{level}] Hint: {hint}")
        print(f"[{level}] Traceback for '{module_name}':")
        print(traceback.format_exc())
        return not required

def verify_installation():
    """Verify installation."""
    try:
        # Check torch
        import torch
        print(f"\nVerification successful.")
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA Available: {torch.cuda.is_available()}")
        verification_ok = True
        
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print("\nChecking compiled modules in dependency order: torch -> flex_gemm -> o_voxel")

            # Required for current Windows wheel stack
            verification_ok &= verify_module_import(
                "flex_gemm",
                required=True,
                hint="If DLL load fails, check CUDA runtime DLL availability and ensure torch imports first."
            )
            verification_ok &= verify_module_import(
                "o_voxel",
                required=True,
                hint="o_voxel depends on flex_gemm; fix flex_gemm import errors first."
            )

            # Optional accelerators / extensions
            verify_module_import("nvdiffrast", required=False)
            verify_module_import(
                "flash_attn",
                required=False,
                hint="If unavailable, set ATTN_BACKEND=sdpa or xformers at runtime."
            )

        # Explicit PIL Check
        try:
            from PIL import Image
            print("[OK] PIL (Pillow) detected.")
        except ImportError:
            print("[ERROR] PIL (Pillow) not found! This is required.")
            return False

        return verification_ok
    except ImportError as e:
        print(f"Verification failed: {str(e)}")
        return False

if __name__ == "__main__":
    install_dependencies()
    if verify_installation():
        print("\nInstallation completed and verified!")
        print("You can now run 'run_app.py' or 'app.py'")
    else:
        print("\nInstallation completed but verification failed.")
        sys.exit(1)
