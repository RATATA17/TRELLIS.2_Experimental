import gradio as gr

import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:512,roundup_power2_divisions:16"
os.environ['TORCH_CUDNN_V8_API_ENABLED'] = '1'
import tempfile
import gc
import multiprocessing as mp
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import shutil
import cv2
from typing import *
import torch
import numpy as np
from PIL import Image
import base64
import io
from trellis2.modules.sparse import SparseTensor
from trellis2.pipelines import Trellis2ImageTo3DPipeline
from trellis2.renderers import EnvMap
from trellis2.utils import render_utils
import o_voxel
import warnings

warnings.filterwarnings('ignore', category=FutureWarning, module='timm')
warnings.filterwarnings('ignore', category=UserWarning, message='.*expandable_segments.*')


_worker_pipeline = None
_worker_envmap = None
_worker_executor_a = None
_worker_executor_b = None
_worker_next_slot = 0


MAX_SEED = np.iinfo(np.int32).max
TMP_DIR = os.path.join(tempfile.gettempdir(), 'trellis2_app_mod_tmp')
MODES = [
    {"name": "Normal", "icon": "assets/app/normal.png", "render_key": "normal"},
    {"name": "Clay render", "icon": "assets/app/clay.png", "render_key": "clay"},
    {"name": "Base color", "icon": "assets/app/basecolor.png", "render_key": "base_color"},
    {"name": "HDRI forest", "icon": "assets/app/hdri_forest.png", "render_key": "shaded_forest"},
    {"name": "HDRI sunset", "icon": "assets/app/hdri_sunset.png", "render_key": "shaded_sunset"},
    {"name": "HDRI courtyard", "icon": "assets/app/hdri_courtyard.png", "render_key": "shaded_courtyard"},
]
STEPS = 8
DEFAULT_MODE = 3
DEFAULT_STEP = 3
DEFAULT_USE_SEPARATE_WORKER = True

APP_USE_SEPARATE_WORKER = DEFAULT_USE_SEPARATE_WORKER


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
    color:
    background-color: var(--color-accent);
    padding: 3px 6px;
    user-select: none;
}

.previewer-container .tips-text {
    position: absolute;
    right: 10px;
    top: 50px;
    color:
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
    border: 2px solid
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
    background:
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
        // We select all elements with class 'previewer-main-image'
        allImgs.forEach(img => img.classList.remove('visible'));

        // 3. Construct the specific ID for the current state
        // Format: view-m{mode}-s{step}
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


def get_user_dir(req: gr.Request) -> str:
    return os.path.join(TMP_DIR, str(req.session_hash))


def get_preview_dir(req: gr.Request) -> str:
    return os.path.join(get_user_dir(req), 'preview')


def reset_preview_dir(req: gr.Request) -> str:
    preview_dir = get_preview_dir(req)
    if os.path.exists(preview_dir):
        shutil.rmtree(preview_dir)
    os.makedirs(preview_dir, exist_ok=True)
    return preview_dir


def get_gradio_file_url(file_path: str) -> str:
    normalized_path = os.path.abspath(file_path).replace('\\', '/')
    return f"/gradio_api/file={normalized_path}"


def _save_latents_to_path(latents: Tuple[SparseTensor, SparseTensor, int], latent_path: str) -> None:
    shape_slat, tex_slat, res = latents
    np.savez(
        latent_path,
        shape_slat_feats=shape_slat.feats.cpu().numpy(),
        tex_slat_feats=tex_slat.feats.cpu().numpy(),
        coords=shape_slat.coords.cpu().numpy(),
        res=np.int32(res),
    )


def _build_pipeline_type(resolution: str) -> str:
    return {
        "512": "512",
        "1024": "1024_cascade",
        "1536": "1536_cascade",
        "2048": "1536_cascade",
    }[resolution]


def _worker_initializer(worker_profile: str) -> None:
    global _worker_pipeline
    global _worker_envmap

    if _worker_pipeline is not None and _worker_envmap is not None:
        return

    _worker_pipeline = Trellis2ImageTo3DPipeline.from_pretrained('microsoft/TRELLIS.2-4B')
    _worker_pipeline.cuda()

    _worker_envmap = {
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


@torch.inference_mode()
def _worker_generate(payload: dict) -> dict:
    global _worker_pipeline
    global _worker_envmap

    image = Image.open(io.BytesIO(payload['image_bytes'])).convert('RGBA')
    outputs, latents = _worker_pipeline.run(
        image,
        seed=payload['seed'],
        preprocess_image=False,
        sparse_structure_sampler_params=payload['sparse_structure_sampler_params'],
        shape_slat_sampler_params=payload['shape_slat_sampler_params'],
        tex_slat_sampler_params=payload['tex_slat_sampler_params'],
        pipeline_type=payload['pipeline_type'],
        return_latent=True,
    )

    mesh = outputs[0]
    mesh.simplify(16777216)

    images = render_utils.render_snapshot(
        mesh,
        resolution=1024,
        r=2,
        fov=36,
        nviews=STEPS,
        envmap=_worker_envmap,
    )

    os.makedirs(payload['preview_dir'], exist_ok=True)
    for m_idx, mode in enumerate(MODES):
        for s_idx in range(STEPS):
            preview_img_path = os.path.join(payload['preview_dir'], f"m{m_idx}_s{s_idx}.jpg")
            Image.fromarray(images[mode['render_key']][s_idx]).convert("RGB").save(
                preview_img_path,
                format="jpeg",
                quality=85,
            )

    _save_latents_to_path(latents, payload['latent_path'])

    del outputs
    del latents
    del mesh
    del images
    gc.collect()
    torch.cuda.empty_cache()

    return {'ok': True}


def _create_worker_executor() -> ProcessPoolExecutor:
    ctx = mp.get_context('spawn')
    return ProcessPoolExecutor(
        max_workers=1,
        mp_context=ctx,
        initializer=_worker_initializer,
        initargs=('performance',),
    )


def _get_worker_executor(slot: int) -> ProcessPoolExecutor:
    global _worker_executor_a
    global _worker_executor_b

    if slot == 0:
        if _worker_executor_a is None:
            _worker_executor_a = _create_worker_executor()
        return _worker_executor_a

    if _worker_executor_b is None:
        _worker_executor_b = _create_worker_executor()
    return _worker_executor_b


def _replace_worker_executor(slot: int) -> None:
    global _worker_executor_a
    global _worker_executor_b

    old_executor = _worker_executor_a if slot == 0 else _worker_executor_b
    if slot == 0:
        _worker_executor_a = None
    else:
        _worker_executor_b = None

    if old_executor is not None:
        try:
            old_executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass


def _submit_with_double_buffer(payload: dict) -> dict:
    global _worker_next_slot

    primary_slot = _worker_next_slot
    fallback_slot = 1 - primary_slot
    _worker_next_slot = fallback_slot

    primary_executor = _get_worker_executor(primary_slot)
    try:
        return primary_executor.submit(_worker_generate, payload).result()
    except Exception as primary_error:
        print(f"Primary worker slot {primary_slot} failed; fallback to slot {fallback_slot}: {primary_error}")
        _replace_worker_executor(primary_slot)

        fallback_executor = _get_worker_executor(fallback_slot)
        try:
            return fallback_executor.submit(_worker_generate, payload).result()
        except Exception as fallback_error:
            print(f"Fallback worker slot {fallback_slot} failed: {fallback_error}")
            _replace_worker_executor(fallback_slot)
            raise RuntimeError(
                f"Both worker executors failed (primary={primary_slot}, fallback={fallback_slot})"
            ) from fallback_error


def _init_double_buffer_executors() -> None:
    _get_worker_executor(0)
    _get_worker_executor(1)


def start_session(req: gr.Request):
    user_dir = get_user_dir(req)
    os.makedirs(user_dir, exist_ok=True)
    
    
def end_session(req: gr.Request):
    user_dir = get_user_dir(req)
    if os.path.exists(user_dir):
        shutil.rmtree(user_dir)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def preprocess_image(image: Image.Image) -> Image.Image:
    """
    Preprocess the input image.

    Args:
        image (Image.Image): The input image.

    Returns:
        Image.Image: The preprocessed image.
    """
    if pipeline is None:
        # separate worker mode: main process does not hold pipeline
        return image
    processed_image = pipeline.preprocess_image(image)
    return processed_image


def get_latent_dir(req: gr.Request) -> str:
    return os.path.join(get_user_dir(req), 'latent')


def get_latent_state_path(req: gr.Request) -> str:
    return os.path.join(get_latent_dir(req), 'current_latent.npz')


def pack_state(latents: Tuple[SparseTensor, SparseTensor, int], req: gr.Request) -> dict:
    latent_dir = get_latent_dir(req)
    os.makedirs(latent_dir, exist_ok=True)

    latent_path = get_latent_state_path(req)
    _save_latents_to_path(latents, latent_path)
    return {
        'latent_path': latent_path,
    }
    
    
def unpack_state(state: dict) -> Tuple[SparseTensor, SparseTensor, int]:
    if 'latent_path' in state:
        with np.load(state['latent_path']) as data:
            shape_slat_feats = data['shape_slat_feats']
            tex_slat_feats = data['tex_slat_feats']
            coords = data['coords']
            res = int(data['res'])
    else:
        # backward compatibility for in-memory state format
        shape_slat_feats = state['shape_slat_feats']
        tex_slat_feats = state['tex_slat_feats']
        coords = state['coords']
        res = int(state['res'])

    shape_slat = SparseTensor(
        feats=torch.from_numpy(shape_slat_feats).cuda(),
        coords=torch.from_numpy(coords).cuda(),
    )
    tex_slat = shape_slat.replace(torch.from_numpy(tex_slat_feats).cuda())
    return shape_slat, tex_slat, res


def get_seed(randomize_seed: bool, seed: int) -> int:
    """
    Get the random seed.
    """
    return np.random.randint(0, MAX_SEED) if randomize_seed else seed


@torch.inference_mode()
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
) -> str:
    torch.cuda.empty_cache()
    preview_dir = reset_preview_dir(req)
    preview_cache_buster = datetime.now().strftime("%Y%m%d%H%M%S%f")
    pipeline_type = _build_pipeline_type(resolution)

    if APP_USE_SEPARATE_WORKER:
        latent_dir = get_latent_dir(req)
        os.makedirs(latent_dir, exist_ok=True)
        latent_path = get_latent_state_path(req)

        image_buffer = io.BytesIO()
        image.save(image_buffer, format='PNG')
        payload = {
            'image_bytes': image_buffer.getvalue(),
            'seed': seed,
            'sparse_structure_sampler_params': {
                "steps": ss_sampling_steps,
                "guidance_strength": ss_guidance_strength,
                "guidance_rescale": ss_guidance_rescale,
                "rescale_t": ss_rescale_t,
            },
            'shape_slat_sampler_params': {
                "steps": shape_slat_sampling_steps,
                "guidance_strength": shape_slat_guidance_strength,
                "guidance_rescale": shape_slat_guidance_rescale,
                "rescale_t": shape_slat_rescale_t,
            },
            'tex_slat_sampler_params': {
                "steps": tex_slat_sampling_steps,
                "guidance_strength": tex_slat_guidance_strength,
                "guidance_rescale": tex_slat_guidance_rescale,
                "rescale_t": tex_slat_rescale_t,
            },
            'pipeline_type': pipeline_type,
            'preview_dir': preview_dir,
            'latent_path': latent_path,
        }

        result = _submit_with_double_buffer(payload)

        if not result.get('ok', False):
            raise RuntimeError(result.get('error', 'Worker generation failed'))

        state = {'latent_path': latent_path}
        images = None
        outputs = None
        mesh = None
    else:
        outputs, latents = pipeline.run(
            image,
            seed=seed,
            preprocess_image=False,
            sparse_structure_sampler_params={
                "steps": ss_sampling_steps,
                "guidance_strength": ss_guidance_strength,
                "guidance_rescale": ss_guidance_rescale,
                "rescale_t": ss_rescale_t,
            },
            shape_slat_sampler_params={
                "steps": shape_slat_sampling_steps,
                "guidance_strength": shape_slat_guidance_strength,
                "guidance_rescale": shape_slat_guidance_rescale,
                "rescale_t": shape_slat_rescale_t,
            },
            tex_slat_sampler_params={
                "steps": tex_slat_sampling_steps,
                "guidance_strength": tex_slat_guidance_strength,
                "guidance_rescale": tex_slat_guidance_rescale,
                "rescale_t": tex_slat_rescale_t,
            },
            pipeline_type=pipeline_type,
            return_latent=True,
        )

        mesh = outputs[0]
        mesh.simplify(16777216)

        images = render_utils.render_snapshot(
            mesh,
            resolution=1024,
            r=2,
            fov=36,
            nviews=STEPS,
            envmap=envmap
        )

        state = pack_state(latents, req)
        del latents
    
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    
    images_html = ""
    for m_idx, mode in enumerate(MODES):
        for s_idx in range(STEPS):
            unique_id = f"view-m{m_idx}-s{s_idx}"
            
            is_visible = (m_idx == DEFAULT_MODE and s_idx == DEFAULT_STEP)
            vis_class = "visible" if is_visible else ""
            loading_mode = "eager" if is_visible else "lazy"
            
            preview_img_path = os.path.join(preview_dir, f"m{m_idx}_s{s_idx}.jpg")
            if images is not None:
                Image.fromarray(images[mode['render_key']][s_idx]).convert("RGB").save(
                    preview_img_path,
                    format="jpeg",
                    quality=85,
                )
            img_url = f"{get_gradio_file_url(preview_img_path)}?v={preview_cache_buster}"
            
            images_html += f"""
                <img id="{unique_id}" 
                     class="previewer-main-image {vis_class}" 
                     src="{img_url}" 
                     loading="{loading_mode}">
            """
    
    btns_html = ""
    for idx, mode in enumerate(MODES):        
        active_class = "active" if idx == DEFAULT_MODE else ""
        btns_html += f"""
            <img src="{mode['icon_base64']}" 
                 class="mode-btn {active_class}" 
                 onclick="selectMode({idx})"
                 title="{mode['name']}">
        """
    
    full_html = f"""
    <div class="previewer-container">
        <div class="tips-wrapper">
            <div class="tips-icon">💡Tips</div>
            <div class="tips-text">
                <p>● <b>Render Mode</b> - Click on the circular buttons to switch between different render modes.</p>
                <p>● <b>View Angle</b> - Drag the slider to change the view angle.</p>
            </div>
        </div>
        
        <!-- Row 1: Viewport containing 48 static <img> tags -->
        <div class="display-row">
            {images_html}
        </div>
        
        <!-- Row 2 -->
        <div class="mode-row" id="btn-group">
            {btns_html}
        </div>

        <!-- Row 3: Slider -->
        <div class="slider-row">
            <input type="range" id="custom-slider" min="0" max="{STEPS - 1}" value="{DEFAULT_STEP}" step="1" oninput="onSliderChange(this.value)">
        </div>
    </div>
    """
    del outputs
    del images
    del mesh
    gc.collect()

    return state, full_html


@torch.inference_mode()
def extract_glb(
    state: dict,
    decimation_target: int,
    texture_size: int,
    req: gr.Request,
    progress=gr.Progress(track_tqdm=True),
) -> Tuple[str, str]:
    """
    Extract a GLB file from the 3D model.

    Args:
        state (dict): The state of the generated 3D model.
        decimation_target (int): The target face count for decimation.
        texture_size (int): The texture resolution.

    Returns:
        str: The path to the extracted GLB file.
    """
    user_dir = get_user_dir(req)
    shape_slat, tex_slat, res = unpack_state(state)
    
    torch.cuda.empty_cache()
    
    mesh = pipeline.decode_latent(shape_slat, tex_slat, res)[0]
    
    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices,
        faces=mesh.faces,
        attr_volume=mesh.attrs,
        coords=mesh.coords,
        attr_layout=pipeline.pbr_attr_layout,
        grid_size=res,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=decimation_target,
        texture_size=texture_size,
        remesh=True,
        remesh_band=1,
        remesh_project=0,
        use_tqdm=True,
    )
    
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%dT%H%M%S") + f".{now.microsecond // 1000:03d}"
    os.makedirs(user_dir, exist_ok=True)
    glb_path = os.path.join(user_dir, f'sample_{timestamp}.glb')
    glb.export(glb_path, extension_webp=False)

    del mesh
    del shape_slat
    del tex_slat
    gc.collect()

    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    
    return glb_path, glb_path


with gr.Blocks(delete_cache=(600, 600)) as demo:
    gr.Markdown("""
    * Upload an image (preferably with an alpha-masked foreground object) and click Generate to create a 3D asset.
    * Click Extract GLB to export and download the generated GLB file if you're satisfied with the result. Otherwise, try another time.
    * **OPTIMIZED VERSION**: Fixed deprecation warnings, improved CUDA memory management, extended texture/resolution options
    """)
    
    with gr.Row():
        with gr.Column(scale=1, min_width=360):
            image_prompt = gr.Image(label="Image Prompt", format="png", image_mode="RGBA", type="pil", height=400)
            
            resolution = gr.Radio(["512", "1024", "1536", "2048"], label="Resolution", value="512")
            seed = gr.Slider(0, MAX_SEED, label="Seed", value=0, step=1)
            randomize_seed = gr.Checkbox(label="Randomize Seed", value=True)
            
            decimation_target = gr.Slider(
                50000, 2000000, 
                label="Decimation Target (Face Count)", 
                value=500000, 
                step=50000,
                info="Recommended: 500k-1M. Higher values may cause instability."
            )
            
            texture_size = gr.Slider(
                512, 8192, 
                label="Texture Size", 
                value=2048, 
                step=512,
                info="Higher values = better texture quality but longer processing"
            )
            
            generate_btn = gr.Button("Generate", variant="primary")

            gr.Markdown("Worker mode is selected at process startup via CLI args.")
                
            with gr.Accordion(label="Advanced Settings", open=False):                
                gr.Markdown("### Stage 1: Sparse Structure Generation")
                with gr.Row():
                    ss_guidance_strength = gr.Slider(1.0, 10.0, label="Guidance Strength", value=7.5, step=0.1)
                    ss_guidance_rescale = gr.Slider(0.0, 1.0, label="Guidance Rescale", value=0.7, step=0.01)
                with gr.Row():
                    ss_sampling_steps = gr.Slider(1, 50, label="Sampling Steps", value=12, step=1)
                    ss_rescale_t = gr.Slider(1.0, 6.0, label="Rescale T", value=5.0, step=0.1)
                    
                gr.Markdown("### Stage 2: Shape Generation")
                with gr.Row():
                    shape_slat_guidance_strength = gr.Slider(1.0, 10.0, label="Guidance Strength", value=7.5, step=0.1)
                    shape_slat_guidance_rescale = gr.Slider(0.0, 1.0, label="Guidance Rescale", value=0.5, step=0.01)
                with gr.Row():
                    shape_slat_sampling_steps = gr.Slider(1, 50, label="Sampling Steps", value=12, step=1)
                    shape_slat_rescale_t = gr.Slider(1.0, 6.0, label="Rescale T", value=3.0, step=0.1)
                    
                gr.Markdown("### Stage 3: Material Generation")
                with gr.Row():
                    tex_slat_guidance_strength = gr.Slider(1.0, 10.0, label="Guidance Strength", value=1.0, step=0.1)
                    tex_slat_guidance_rescale = gr.Slider(0.0, 1.0, label="Guidance Rescale", value=0.0, step=0.01)
                with gr.Row():
                    tex_slat_sampling_steps = gr.Slider(1, 50, label="Sampling Steps", value=12, step=1)
                    tex_slat_rescale_t = gr.Slider(1.0, 6.0, label="Rescale T", value=3.0, step=0.1)                

        with gr.Column(scale=10):
            with gr.Walkthrough(selected=0) as walkthrough:
                with gr.Step("Preview", id=0):
                    preview_output = gr.HTML(empty_html, label="3D Asset Preview", show_label=True, container=True)
                    extract_btn = gr.Button("Extract GLB", variant="primary")
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
        queue=False,
    ).then(
        lambda: gr.Walkthrough(selected=0), 
        outputs=walkthrough,
        queue=False,
    ).then(
        image_to_3d,
        inputs=[
            image_prompt, seed, resolution,
            ss_guidance_strength, ss_guidance_rescale, ss_sampling_steps, ss_rescale_t,
            shape_slat_guidance_strength, shape_slat_guidance_rescale, shape_slat_sampling_steps, shape_slat_rescale_t,
            tex_slat_guidance_strength, tex_slat_guidance_rescale, tex_slat_sampling_steps, tex_slat_rescale_t,
        ],
        outputs=[output_buf, preview_output],
        queue=False,
    )
    
    extract_btn.click(
        lambda: gr.Walkthrough(selected=1), 
        outputs=walkthrough,
        queue=False,
    ).then(
        extract_glb,
        inputs=[output_buf, decimation_target, texture_size],
        outputs=[glb_output, download_btn],
        queue=False,
    )
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--worker-backend", choices=["separate", "inprocess"], default="separate")
    parser.add_argument("--worker-refresh-interval", type=int, default=10)
    args, _unknown = parser.parse_known_args()

    APP_USE_SEPARATE_WORKER = args.worker_backend == "separate"

    os.makedirs(TMP_DIR, exist_ok=True)

    btn_img_base64_strs = {}
    for i in range(len(MODES)):
        icon = Image.open(MODES[i]['icon'])
        MODES[i]['icon_base64'] = image_to_base64(icon)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    
    if APP_USE_SEPARATE_WORKER:
        print("Worker backend: separate (double-buffer executors)")
        pipeline = None
        envmap = None
        _init_double_buffer_executors()
    else:
        print("Worker backend: inprocess")
        print("Loading TRELLIS.2 pipeline...")
        pipeline = Trellis2ImageTo3DPipeline.from_pretrained('microsoft/TRELLIS.2-4B')
        pipeline.cuda()

        print("Loading environment maps...")
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
    
    print("Starting Gradio interface...")
    demo.launch(css=css, head=head, allowed_paths=[TMP_DIR])
