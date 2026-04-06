# pipeline_worker.py
# Runs the TRELLIS.2 pipeline in a separate process to avoid GIL contention with Gradio.
import multiprocessing as mp
import traceback
import os
import sys
import time

from trellis2.utils.log_utils import should_log, log


def _apply_patches():
    """Apply Windows compatibility patches for flex_gemm."""
    import torch

    try:
        import flex_gemm.ops.spconv as spconv
        from flex_gemm.ops.spconv import Algorithm
        # Force the algorithm to EXPLICIT_GEMM
        # This bypasses the 'kernels.triton' error by using standard Torch Matrix Multiplication
        spconv.ALGORITHM = Algorithm.EXPLICIT_GEMM
        log("normal", "[WORKER] flex_gemm EXPLICIT_GEMM patch applied.")
    except ImportError:
        log("normal", "[WORKER] Could not patch flex_gemm spconv.")
    except Exception as e:
        log("normal", f"[WORKER] flex_gemm spconv patch failed: {e}")

    try:
        import flex_gemm.kernels as _fgk
        triton_obj = getattr(_fgk, 'triton', None)
        triton_has_fwd = hasattr(triton_obj, 'indice_weighed_sum_fwd') if triton_obj is not None else False
        triton_has_bwd = hasattr(triton_obj, 'indice_weighed_sum_bwd_input') if triton_obj is not None else False
        log("verbose", (
            "[WORKER][DIAG] flex_gemm.kernels.triton "
            f"exists={hasattr(_fgk, 'triton')} "
            f"is_none={triton_obj is None} "
            f"has_fwd={triton_has_fwd} has_bwd={triton_has_bwd}"
        ))
        needs_fallback = (not hasattr(_fgk, 'triton')) or (triton_obj is None) or (not triton_has_fwd) or (not triton_has_bwd)
        if needs_fallback:
            class _TritonFallback:
                @staticmethod
                def indice_weighed_sum_fwd(feats, indices, weights):
                    N = feats.shape[0]
                    idx = indices.long().clamp(min=0, max=N - 1)  # [M, 8]
                    
                    M_shape, K = idx.shape
                    C = feats.shape[-1]
                    
                    # Accumulate sequentially to avoid a massive [M, K, C] memory spike
                    out = torch.zeros((M_shape, C), dtype=feats.dtype, device=feats.device)
                    for i in range(K):
                        out += feats[idx[:, i]] * weights[:, i].unsqueeze(-1)
                    return out

                @staticmethod
                def indice_weighed_sum_bwd_input(grad_output, indices, weights, N):
                    M, C = grad_output.shape
                    idx = indices.long().clamp(min=0, max=N - 1)
                    weighted_grad = grad_output.unsqueeze(1) * weights.unsqueeze(-1)  # [M, 8, C]
                    grad_feats = torch.zeros(N, C, device=grad_output.device, dtype=grad_output.dtype)
                    grad_feats.scatter_add_(0, idx.reshape(-1, 1).expand(-1, C), weighted_grad.reshape(-1, C))
                    return grad_feats

            _fgk.triton = _TritonFallback()
            log("normal", "[WORKER] flex_gemm Triton fallback patch applied.")
        else:
            log("verbose", "[WORKER][DIAG] Triton fallback skipped because attribute exists.")
    except ImportError:
        log("normal", "[WORKER] Could not patch flex_gemm triton fallback.")
    except Exception as e:
        log("normal", f"[WORKER] flex_gemm triton patch failed: {e}")


def _worker_main(cmd_queue, result_queue):
    """Worker process main loop. Owns the pipeline and all GPU resources."""
    os.environ["TORCHDYNAMO_DISABLE"] = "1"
    base_alloc_conf = "max_split_size_mb:512,roundup_power2_divisions:16"
    # NOTE: expandable_segments is not supported on Windows in current PyTorch builds.
    if os.name == "nt":
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = base_alloc_conf
    else:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = f"expandable_segments:True,{base_alloc_conf}"
    os.environ['TORCH_CUDNN_V8_API_ENABLED'] = '1'
    os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
    # NOTE: log level defaults to normal unless TRELLIS_LOG_LEVEL or TRELLIS_RENDER_DIAG is set.

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    _apply_patches()

    import torch
    import cv2
    import numpy as np
    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    from trellis2.renderers import EnvMap
    from trellis2.utils import render_utils
    from trellis2.modules.sparse import SparseTensor
    import o_voxel

    def _safe_cuda_sync(label: str):
        if not should_log("verbose"):
            return
        try:
            torch.cuda.synchronize()
            log("verbose", f"[WORKER][DIAG] cuda_sync_ok label={label}")
        except Exception as e:
            log("verbose", f"[WORKER][DIAG] cuda_sync_error label={label} err={e}")

    def _tensor_diag(name: str, x: torch.Tensor):
        if not should_log("verbose"):
            return
        try:
            if not isinstance(x, torch.Tensor):
                log("verbose", f"[WORKER][DIAG] {name}: not_tensor type={type(x)}")
                return
            shape = tuple(x.shape)
            dtype = str(x.dtype)
            device = str(x.device)
            finite = torch.isfinite(x).all().item() if x.is_floating_point() else True
            has_nan = torch.isnan(x).any().item() if x.is_floating_point() else False
            has_inf = torch.isinf(x).any().item() if x.is_floating_point() else False
            log("verbose", (
                f"[WORKER][DIAG] {name}: shape={shape} dtype={dtype} device={device} "
                f"finite={finite} nan={has_nan} inf={has_inf}"
            ))
        except Exception as e:
            log("verbose", f"[WORKER][DIAG] {name}: diag_failed err={e}")

    def _array_diag(name: str, x):
        if not should_log("debug"):
            return
        try:
            arr = np.asarray(x)
            finite = np.isfinite(arr).all() if np.issubdtype(arr.dtype, np.floating) else True
            vmin = float(arr.min()) if arr.size else float('nan')
            vmax = float(arr.max()) if arr.size else float('nan')
            mean = float(arr.mean()) if arr.size else float('nan')
            std = float(arr.std()) if arr.size else float('nan')
            log("debug", (
                f"[WORKER][DIAG] {name}: shape={arr.shape} dtype={arr.dtype} finite={finite} "
                f"min={vmin:.6g} max={vmax:.6g} mean={mean:.6g} std={std:.6g}"
            ))
        except Exception as e:
            log("debug", f"[WORKER][DIAG] {name}: diag_failed err={e}")

    def _envmap_diag(stage: str):
        if not should_log("verbose"):
            return
        try:
            for env_name, env in envmap.items():
                has_image = hasattr(env, "image")
                has_backend = hasattr(env, "_nvdiffrec_envlight")
                has_cubemap_cpu = hasattr(env, "_cubemap_cpu")
                log("verbose", (
                    "[WORKER][DIAG] "
                    f"envmap_state stage={stage} name={env_name} "
                    f"has_image={has_image} has_backend={has_backend} has_cubemap_cpu={has_cubemap_cpu}"
                ))
                if has_image:
                    _tensor_diag(f"envmap.{env_name}.image", getattr(env, "image", None))
                if has_backend:
                    backend = getattr(env, "_nvdiffrec_envlight", None)
                    if backend is not None and hasattr(backend, "base"):
                        _tensor_diag(f"envmap.{env_name}.backend.base", backend.base)
                if has_cubemap_cpu:
                    _tensor_diag(f"envmap.{env_name}._cubemap_cpu", getattr(env, "_cubemap_cpu", None))
        except Exception as e:
            log("verbose", f"[WORKER][DIAG] envmap_state stage={stage} diag_failed err={e}")

    log("normal", "[WORKER] Loading pipeline...")
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained('microsoft/TRELLIS.2-4B')
    pipeline.cuda()

    worker_started_at = time.time()
    generation_count = 0
    log("normal", (
        "[WORKER][INIT] "
        f"pid={os.getpid()} "
        f"torch={torch.__version__} "
        f"cuda={torch.version.cuda} "
        f"cudnn={torch.backends.cudnn.version()} "
        f"device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'} "
        f"alloc_conf={os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '')}"
    ))

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

    for env_name, env in envmap.items():
        # EnvMap holds source tensor in `image` until backend is materialized.
        _tensor_diag(f"envmap.{env_name}.image", getattr(env, "image", None))
    _envmap_diag("init")

    log("normal", "[WORKER] Pipeline ready.")
    result_queue.put({"status": "ready"})

    while True:
        cmd = cmd_queue.get()
        action = cmd["action"]

        if action == "shutdown":
            log("normal", "[WORKER] Shutting down.")
            break

        elif action == "preprocess":
            try:
                image = pipeline.preprocess_image(cmd["image"])
                result_queue.put({"status": "ok", "image": image})
            except Exception as e:
                traceback.print_exc()
                result_queue.put({"status": "error", "error": str(e)})

        elif action == "generate":
            try:
                generation_count += 1
                log("normal", (
                    "[WORKER][GEN] "
                    f"count={generation_count} "
                    f"uptime_sec={time.time() - worker_started_at:.1f} "
                    f"seed={cmd['seed']} "
                    f"pipeline_type={cmd['pipeline_type']} "
                    f"nviews={cmd['nviews']}"
                ))

                from tools.profiling_wrapper import AuraProfiler
                from tools.sync_hunter import hunt_syncs
                prof_cfg = cmd["profiling"]
                prof_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tmp', 'profiling')
                master_enabled = prof_cfg["enable_python"] or prof_cfg["enable_torch"]
                profiler = AuraProfiler(
                    log_dir=prof_dir,
                    actor_name="",
                    enabled=master_enabled,
                    enable_python=prof_cfg["enable_python"],
                    enable_torch=prof_cfg["enable_torch"],
                    schedule_config={"wait": 0, "warmup": 0, "active": 10000, "repeat": 0},
                    delay_sec=prof_cfg["delay_sec"],
                    max_duration_sec=prof_cfg["max_duration_sec"],
                    max_events=prof_cfg["max_events"],
                )
                with hunt_syncs(enabled=prof_cfg["enable_sync_hunter"],
                                log_file=os.path.join(prof_dir, "sync_report.txt")):
                    profiler.start()
                    outputs, latents = pipeline.run(
                        cmd["image"],
                        seed=cmd["seed"],
                        preprocess_image=False,
                        sparse_structure_sampler_params=cmd["ss_params"],
                        shape_slat_sampler_params=cmd["shape_params"],
                        tex_slat_sampler_params=cmd["tex_params"],
                        pipeline_type=cmd["pipeline_type"],
                        return_latent=True,
                    )
                    profiler.step()
                profiler.stop_and_save("inference_run")
                _safe_cuda_sync("after_pipeline_run")
                def _vram_mb():
                    return torch.cuda.memory_allocated() / (1024**2)
                log("verbose", f"[VRAM] After pipeline.run: allocated={_vram_mb():.0f}MB")
                mesh = outputs[0]
                log("verbose", f"[WORKER][DIAG] mesh.type={type(mesh).__name__}")
                _tensor_diag("mesh.vertices", mesh.vertices)
                _tensor_diag("mesh.faces", mesh.faces)
                _tensor_diag("mesh.attrs", mesh.attrs)
                _tensor_diag("mesh.coords", mesh.coords)
                
                # Move latents to CPU BEFORE rendering to free VRAM
                shape_slat, tex_slat, res = latents
                _tensor_diag("latent.shape_slat.feats", shape_slat.feats)
                _tensor_diag("latent.tex_slat.feats", tex_slat.feats)
                _tensor_diag("latent.coords", shape_slat.coords)
                state = {
                    'shape_slat_feats': shape_slat.feats.cpu().numpy(),
                    'tex_slat_feats': tex_slat.feats.cpu().numpy(),
                    'coords': shape_slat.coords.cpu().numpy(),
                    'res': res,
                }
                del latents, shape_slat, tex_slat, outputs
                log("verbose", f"[VRAM] After latent offload: allocated={_vram_mb():.0f}MB")
                torch.cuda.empty_cache()
                
                mesh.simplify(16777216)  # nvdiffrast limit
                log("verbose", f"[VRAM] After simplify: allocated={_vram_mb():.0f}MB")
                _envmap_diag(f"pre_render_gen{generation_count}")
                
                with torch.inference_mode():
                    images = render_utils.render_snapshot(
                        mesh, 
                        resolution=512,
                        r=2, fov=36,
                        nviews=cmd["nviews"], 
                        envmap=envmap
                    )
                _safe_cuda_sync("after_render_snapshot")
                _envmap_diag(f"post_render_gen{generation_count}")
                for mode_key in ["normal", "clay", "base_color", "shaded_forest", "shaded_sunset", "shaded_courtyard"]:
                    if mode_key in images and len(images[mode_key]) > 0:
                        _array_diag(f"render.{mode_key}[0]", images[mode_key][0])
                log("verbose", f"[VRAM] After render: allocated={_vram_mb():.0f}MB")
                torch.cuda.empty_cache()
                result_queue.put({"status": "ok", "state": state, "images": images})
                del mesh, images, state
                torch.cuda.empty_cache()
            except Exception as e:
                traceback.print_exc()
                result_queue.put({"status": "error", "error": str(e)})
                torch.cuda.empty_cache()

        elif action == "extract_glb":
            try:
                _envmap_diag("pre_extract_offload")
                # Offload envmaps to CPU — not used during GLB export
                for env in envmap.values():
                    env.offload()
                _envmap_diag("post_extract_offload")
                torch.cuda.empty_cache()
                
                state = cmd["state"]
                shape_slat = SparseTensor(
                    feats=torch.from_numpy(state['shape_slat_feats']).cuda(),
                    coords=torch.from_numpy(state['coords']).cuda(),
                )
                tex_slat = shape_slat.replace(torch.from_numpy(state['tex_slat_feats']).cuda())
                res = state['res']
                
                mesh = pipeline.decode_latent(shape_slat, tex_slat, res)[0]
                mesh.attrs = mesh.attrs.float()
                
                # Free everything possible before heavy GLB postprocessing
                del shape_slat, tex_slat, state
                for name, model in pipeline.models.items():
                    model.cpu()
                torch.cuda.empty_cache()
                
                glb = o_voxel.postprocess.to_glb(
                    vertices=mesh.vertices,
                    faces=mesh.faces,
                    attr_volume=mesh.attrs,
                    coords=mesh.coords,
                    attr_layout=pipeline.pbr_attr_layout,
                    grid_size=res,
                    aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                    decimation_target=cmd["decimation_target"],
                    texture_size=cmd["texture_size"],
                    remesh=True,
                    remesh_band=1,
                    remesh_project=0,
                    use_tqdm=True,
                )
                glb.export(cmd["glb_path"], extension_webp=False)
                del mesh, glb
                torch.cuda.empty_cache()
                # Flush any lingering CUDA errors (e.g. from xatlas assertions)
                # so they don't poison subsequent kernel launches
                torch.cuda.synchronize()
                
                # Reload envmaps back to GPU for next render cycle
                for env in envmap.values():
                    env.reload()
                _envmap_diag("post_extract_reload")
                
                result_queue.put({"status": "ok", "glb_path": cmd["glb_path"]})
            except Exception as e:
                traceback.print_exc()
                # Reload envmaps even on failure, so next generate's render works
                for env in envmap.values():
                    try:
                        env.reload()
                    except Exception:
                        pass
                _envmap_diag("extract_fail_post_reload")
                torch.cuda.empty_cache()
                result_queue.put({"status": "error", "error": str(e)})


class PipelineWorker:
    """Manages a subprocess that owns the GPU pipeline."""

    def __init__(self):
        ctx = mp.get_context('spawn')  # 'spawn' is required for CUDA on Windows
        self.cmd_queue = ctx.Queue()
        self.result_queue = ctx.Queue()
        self.process = ctx.Process(target=_worker_main, args=(self.cmd_queue, self.result_queue))
        self.process.daemon = True
        log("normal", "[INFO] Starting pipeline worker process...")
        self.process.start()
        msg = self.result_queue.get(timeout=300)
        assert msg["status"] == "ready", f"Worker failed to start: {msg}"
        log("normal", "[INFO] Pipeline worker ready.")

    def preprocess(self, image):
        self.cmd_queue.put({"action": "preprocess", "image": image})
        result = self.result_queue.get(timeout=120)
        if result["status"] == "error":
            raise RuntimeError(result["error"])
        return result["image"]

    def generate(self, image, seed, ss_params, shape_params, tex_params, pipeline_type, nviews, profiling=None):
        self.cmd_queue.put({
            "action": "generate",
            "image": image,
            "seed": seed,
            "ss_params": ss_params,
            "shape_params": shape_params,
            "tex_params": tex_params,
            "pipeline_type": pipeline_type,
            "nviews": nviews,
            "profiling": profiling or {},
        })
        result = self.result_queue.get(timeout=600)
        if result["status"] == "error":
            raise RuntimeError(result["error"])
        return result["state"], result["images"]

    def extract_glb(self, state, decimation_target, texture_size, glb_path):
        self.cmd_queue.put({
            "action": "extract_glb",
            "state": state,
            "decimation_target": decimation_target,
            "texture_size": texture_size,
            "glb_path": glb_path,
        })
        result = self.result_queue.get(timeout=600)
        if result["status"] == "error":
            raise RuntimeError(result["error"])
        return result["glb_path"]

    def shutdown(self):
        try:
            self.cmd_queue.put({"action": "shutdown"})
            self.process.join(timeout=10)
        except Exception:
            self.process.kill()
