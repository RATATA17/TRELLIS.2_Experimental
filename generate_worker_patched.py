import sys
import os
import json
import base64
import io
import gc
from datetime import datetime
import torch
from PIL import Image
from trellis2.modules.sparse import SparseTensor
from trellis2.pipelines import Trellis2ImageTo3DPipeline
from trellis2.representations import Mesh
from trellis2.utils import render_utils


def _decode_image(b64_str: str) -> Image.Image:
    raw = base64.b64decode(b64_str.encode("ascii"))
    return Image.open(io.BytesIO(raw)).convert("RGBA")

def _write_progress(path: str, stage: str, pct: int) -> None:
    if not path:
        return
    tmp = path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"stage": stage, "pct": int(pct)}, f)
        os.replace(tmp, path)
    except Exception:
        pass

def _log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: generate_worker_patched.py <json_args>", file=sys.stderr)
        return 2

    _log("[WORKER] start")
    arg = sys.argv[1]
    if os.path.exists(arg):
        with open(arg, "r", encoding="utf-8") as f:
            args = json.load(f)
    else:
        args = json.loads(arg)
    image_b64 = args["image_b64"]
    seed = int(args["seed"])
    resolution = args["resolution"]
    out_dir = args["out_dir"]
    preview_resolution = int(args["preview_resolution"])
    preview_views = int(args["preview_views"])
    progress_path = args.get("progress_path", "")

    ss = args["ss"]
    shape = args["shape"]
    tex = args["tex"]

    os.makedirs(out_dir, exist_ok=True)
    image = _decode_image(image_b64)

    with torch.no_grad():
        _log("pipeline_load:start")
        pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
        _log("pipeline_load:end")
        pipeline.cuda()
        _write_progress(progress_path, "sampling_sparse", 0)
        _log("sampling:start")
        outputs, latents = pipeline.run(
            image,
            seed=seed,
            preprocess_image=False,
            sparse_structure_sampler_params=ss,
            shape_slat_sampler_params=shape,
            tex_slat_sampler_params=tex,
            pipeline_type={
                "512": "512",
                "1024": "1024_cascade",
                "1536": "1536_cascade",
            }[resolution],
            return_latent=True,
        )
        _log("sampling:end")
        _write_progress(progress_path, "sampling_done", 70)

        mesh = outputs[0]
        mesh.simplify(16777216)  # nvdiffrast limit

        preview_mesh = Mesh(mesh.vertices, mesh.faces)
        _write_progress(progress_path, "preview_render", 85)
        _log("preview_render:start")
        images = render_utils.render_snapshot(
            preview_mesh,
            resolution=preview_resolution,
            r=2,
            fov=36,
            nviews=preview_views,
            return_types=["normal"],
        )
        _log("preview_render:end")
        _write_progress(progress_path, "preview_done", 95)

    # Save latents to disk
    state_path = os.path.join(out_dir, "latent_state.pt")
    shape_slat, tex_slat, res = latents
    torch.save(
        {
            "shape_feats": shape_slat.feats.detach().cpu(),
            "tex_feats": tex_slat.feats.detach().cpu(),
            "coords": shape_slat.coords.detach().cpu(),
            "res": int(res),
        },
        state_path,
    )
    _log("state_save:end")
    _write_progress(progress_path, "state_saved", 98)

    # Save preview images
    preview_paths = []
    for idx in range(preview_views):
        img = Image.fromarray(images["normal"][idx])
        path = os.path.join(out_dir, f"preview_normal_{idx}.png")
        img.save(path)
        preview_paths.append(path)
    _log("preview_save:end")
    _write_progress(progress_path, "preview_saved", 100)

    # Cleanup
    try:
        del outputs, latents, mesh, preview_mesh, images, pipeline
    except Exception:
        pass
    torch.cuda.empty_cache()
    gc.collect()

    print(json.dumps({"state_path": state_path, "preview_paths": preview_paths}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
