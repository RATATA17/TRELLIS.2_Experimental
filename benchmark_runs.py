import argparse
import base64
import io
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from statistics import mean, pstdev

from PIL import Image


TS_FMT = "%Y-%m-%d %H:%M:%S"


def _read_last_json_line(path: str):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
        if not lines:
            return None
        return json.loads(lines[-1])
    except Exception:
        return None


def _parse_stage_times(path: str):
    # Extract timestamps from worker log: "[YYYY-MM-DD HH:MM:SS] message"
    times = {}
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                m = re.match(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]\s+(.*)$", line.strip())
                if not m:
                    continue
                ts = datetime.strptime(m.group(1), TS_FMT)
                msg = m.group(2)
                if msg in (
                    "load_state:start",
                    "load_state:end",
                    "pipeline_load:start",
                    "pipeline_load:end",
                    "decode_latent:start",
                    "decode_latent:end",
                    "to_glb:start",
                    "to_glb:end type=<class 'trimesh.base.Trimesh'>",
                    "export:start",
                    "export:end",
                ):
                    times[msg] = ts
    except Exception:
        return {}

    def _dur(a, b):
        if a in times and b in times:
            return (times[b] - times[a]).total_seconds()
        return None

    return {
        "load_state_s": _dur("load_state:start", "load_state:end"),
        "pipeline_load_s": _dur("pipeline_load:start", "pipeline_load:end"),
        "decode_latent_s": _dur("decode_latent:start", "decode_latent:end"),
        "to_glb_s": _dur("to_glb:start", "to_glb:end type=<class 'trimesh.base.Trimesh'>"),
        "export_s": _dur("export:start", "export:end"),
    }


def _encode_image(path: str) -> str:
    img = Image.open(path).convert("RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _run_worker(cmd, log_path, env):
    start = time.time()
    with open(log_path, "w", encoding="utf-8") as logf:
        proc = subprocess.run(
            cmd,
            stdout=logf,
            stderr=subprocess.STDOUT,
            text=True,
            stdin=subprocess.DEVNULL,
            env=env,
            creationflags=0x08000000 if os.name == "nt" else 0,
        )
    return proc.returncode, time.time() - start


def _summarize(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return {
        "min": min(vals),
        "max": max(vals),
        "mean": mean(vals),
        "std": pstdev(vals) if len(vals) > 1 else 0.0,
    }


def main():
    ap = argparse.ArgumentParser(description="Benchmark generate/extract runs.")
    ap.add_argument("--image", required=True, help="Path to input image")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--resolution", default="512")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--decimation_target", type=int, default=200000)
    ap.add_argument("--texture_size", type=int, default=1024)
    ap.add_argument("--remesh", action="store_true", default=True)
    ap.add_argument("--no-extract", action="store_true", default=False)
    ap.add_argument("--out-dir", default=None, help="Output directory")
    args = ap.parse_args()

    base_dir = args.out_dir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "tmp",
        f"bench_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    os.makedirs(base_dir, exist_ok=True)

    image_b64 = _encode_image(args.image)
    results = []

    py = sys.executable
    gen_worker = os.path.join(os.path.dirname(os.path.abspath(__file__)), "generate_worker_patched.py")
    ext_worker = os.path.join(os.path.dirname(os.path.abspath(__file__)), "extract_worker_patched.py")

    env = dict(os.environ)
    env["TQDM_DISABLE"] = "1"

    for i in range(1, args.runs + 1):
        run_dir = os.path.join(base_dir, f"run_{i:02d}")
        os.makedirs(run_dir, exist_ok=True)
        gen_log = os.path.join(run_dir, "gen_worker.log")
        ext_log = os.path.join(run_dir, "extract_worker.log")

        payload = {
            "image_b64": image_b64,
            "seed": args.seed,
            "resolution": args.resolution,
            "out_dir": run_dir,
            "preview_resolution": 512,
            "preview_views": 8,
            "ss": {
                "steps": 12,
                "guidance_strength": 7.5,
                "guidance_rescale": 0.7,
                "rescale_t": 5.0,
            },
            "shape": {
                "steps": 12,
                "guidance_strength": 7.5,
                "guidance_rescale": 0.5,
                "rescale_t": 3.0,
            },
            "tex": {
                "steps": 12,
                "guidance_strength": 1.0,
                "guidance_rescale": 0.0,
                "rescale_t": 3.0,
            },
            "progress_path": os.path.join(run_dir, "gen_progress.json"),
        }
        payload_path = os.path.join(run_dir, "gen_payload.json")
        with open(payload_path, "w", encoding="utf-8") as f:
            json.dump(payload, f)

        gen_rc, gen_s = _run_worker([py, gen_worker, payload_path], gen_log, env)
        gen_json = _read_last_json_line(gen_log)

        state_path = gen_json["state_path"] if isinstance(gen_json, dict) and "state_path" in gen_json else None

        ext_rc = None
        ext_s = None
        stage_times = None
        glb_path = None
        if not args.no_extract and state_path and os.path.exists(state_path):
            ext_payload = {
                "state_path": state_path,
                "out_dir": run_dir,
                "decimation_target": args.decimation_target,
                "texture_size": args.texture_size,
                "remesh": bool(args.remesh),
                "progress_path": os.path.join(run_dir, "extract_progress.json"),
            }
            ext_rc, ext_s = _run_worker([py, ext_worker, json.dumps(ext_payload)], ext_log, env)
            stage_times = _parse_stage_times(ext_log)
            # last non-empty line is glb path
            try:
                with open(ext_log, "r", encoding="utf-8", errors="replace") as f:
                    lines = [line.strip() for line in f.readlines() if line.strip()]
                if lines:
                    glb_path = lines[-1]
            except Exception:
                pass

        result = {
            "run": i,
            "gen_rc": gen_rc,
            "gen_time_s": gen_s,
            "state_path": state_path,
            "ext_rc": ext_rc,
            "ext_time_s": ext_s,
            "glb_path": glb_path,
            "stage_times": stage_times,
            "gen_log": gen_log,
            "ext_log": ext_log if ext_rc is not None else None,
        }
        results.append(result)

    # Summaries
    summary = {
        "gen_time_s": _summarize([r["gen_time_s"] for r in results]),
        "ext_time_s": _summarize([r["ext_time_s"] for r in results if r["ext_time_s"] is not None]),
    }

    summary_path = os.path.join(base_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"results": results, "summary": summary}, f, indent=2)

    # Human-readable summary
    lines = []
    lines.append(f"Benchmark runs: {args.runs}")
    lines.append(f"Image: {args.image}")
    lines.append("")
    for r in results:
        lines.append(f"Run {r['run']:02d}: gen={r['gen_time_s']:.1f}s rc={r['gen_rc']} ext={r['ext_time_s']:.1f}s rc={r['ext_rc']}")
        if r["stage_times"]:
            st = r["stage_times"]
            lines.append(f"  stages: load={st.get('load_state_s')}s, load_model={st.get('pipeline_load_s')}s, decode={st.get('decode_latent_s')}s, to_glb={st.get('to_glb_s')}s, export={st.get('export_s')}s")
    lines.append("")
    if summary["gen_time_s"]:
        g = summary["gen_time_s"]
        lines.append(f"GEN: min={g['min']:.1f}s max={g['max']:.1f}s mean={g['mean']:.1f}s std={g['std']:.1f}s")
    if summary["ext_time_s"]:
        e = summary["ext_time_s"]
        lines.append(f"EXT: min={e['min']:.1f}s max={e['max']:.1f}s mean={e['mean']:.1f}s std={e['std']:.1f}s")

    summary_txt = os.path.join(base_dir, "summary.txt")
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(summary_txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
