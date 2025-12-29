import json
import sys
import os
import torch
import gc
from datetime import datetime
from trellis2.modules.sparse import SparseTensor
from trellis2.pipelines import Trellis2ImageTo3DPipeline
import o_voxel


def _log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


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


def main():
    _log("pool_start")
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
    pipeline.cuda()
    _log("pipeline_loaded")
    for line in sys.stdin:
        if not line.strip():
            continue
        args = json.loads(line)
        state_path = args["state_path"]
        out_dir = args["out_dir"]
        decimation_target = int(args["decimation_target"])
        texture_size = int(args["texture_size"])
        remesh = bool(args.get("remesh", False))
        progress_path = args.get("progress_path", "")

        _write_progress(progress_path, "load_state", 5)
        data = torch.load(state_path, map_location="cpu")
        shape_slat = SparseTensor(feats=data["shape_feats"].cuda(), coords=data["coords"].cuda())
        tex_slat = shape_slat.replace(data["tex_feats"].cuda())
        res = int(data["res"])
        _write_progress(progress_path, "load_state_done", 15)

        with torch.no_grad():
            _write_progress(progress_path, "decode_latent", 45)
            mesh = pipeline.decode_latent(shape_slat, tex_slat, res)[0]
            _write_progress(progress_path, "decode_done", 55)

            _write_progress(progress_path, "to_glb", 70)
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
                remesh=remesh,
                remesh_band=1,
                remesh_project=0,
                use_tqdm=True,
            )
            _write_progress(progress_path, "to_glb_done", 90)

        _write_progress(progress_path, "export", 95)
        timestamp = datetime.now().strftime("%Y-%m-%dT%H%M%S") + ".glb"
        os.makedirs(out_dir, exist_ok=True)
        glb_path = os.path.join(out_dir, f"sample_{timestamp}")
        glb.export(glb_path, extension_webp=False)
        _write_progress(progress_path, "export_done", 100)

        try:
            del mesh, glb, shape_slat, tex_slat
        except Exception:
            pass
        torch.cuda.empty_cache()
        gc.collect()

        print(glb_path, flush=True)
        print("__END__", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
