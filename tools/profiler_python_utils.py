# src/world_model/utils/profiler_python_utils.py
import json
import os
import logging
from collections import defaultdict, Counter

def _recursive_walk(node, registry):
    func_name = node.get('function', '<unknown>')
    # Robust path lookup: try short, then full, then empty
    file_path = node.get('file_path_short') or node.get('file_path') or ''
    
    # Filter Logic: Identify User Code
    norm_path = file_path.replace('\\', '/')
    
    # [ADJUSTMENT] Broader library detection to filter out Ray internals
    is_library = any(x in norm_path for x in ["site-packages", "dist-packages", "lib/python", "ray/"])
    
    # [ADJUSTMENT] Explicitly whitelist your project folders
    # This ensures files in 'rl_agent', 'world_model', or root 'AURA' folders are kept
    is_project = any(x in norm_path for x in ["world_model", "rl_agent", "src/", "AURA"])
    
    # Final logic: It is user code if it is in the project OR (it is not a library AND not python internals)
    is_user_code = is_project or (not is_library and "python" not in norm_path.lower())

    key = f"{func_name}::{file_path}"
    
    time_inclusive = node.get('time', 0.0)
    children_time = sum(c.get('time', 0.0) for c in node.get('children', []))
    time_exclusive = max(0.0, time_inclusive - children_time)
    
    if key not in registry:
        registry[key] = {
            'function': func_name,
            'file': file_path,
            'total_time': 0.0,
            'self_time': 0.0,
            'calls': 0,
            'is_user_code': is_user_code
        }
    
    registry[key]['total_time'] += time_inclusive
    registry[key]['self_time'] += time_exclusive
    registry[key]['calls'] += 1

    for child in node.get('children', []):
        _recursive_walk(child, registry)

def process_profile_json(json_path: str, output_txt_path: str):
    """
    Reads a PyInstrument JSON file, aggregates stats, and writes a summary.
    """
    if not os.path.exists(json_path):
        logging.error(f"Profiler JSON not found: {json_path}")
        return

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            content = f.read()
            if not content:
                logging.error("Profiler JSON file is empty.")
                return
            data = json.loads(content)
            
        root_frame = data.get('root_frame', data)
        registry = {}
        _recursive_walk(root_frame, registry)
        
        # Filter for User Code
        stats_list = [s for s in registry.values() if s['is_user_code']]
        stats_list.sort(key=lambda x: x['total_time'], reverse=True)
        
        with open(output_txt_path, 'w', encoding='utf-8') as f:
            f.write(f"--- AURA PERFORMANCE SUMMARY ---\n")
            f.write(f"Source: {json_path}\n")
            f.write(f"FUNCTION | TOTAL (s) | SELF (s) | CALLS | LOCATION\n")
            f.write("---\n")
            
            for item in stats_list[:50]:
                func = item['function']
                total = f"{item['total_time']:.4f}"
                self_t = f"{item['self_time']:.4f}"
                calls = item['calls']
                loc = item['file']
                # Compact format: No fixed width padding
                f.write(f"{func} | {total} | {self_t} | {calls} | {loc}\n")
                
        logging.info(f">>> AURA PROFILER: Summary generated at {output_txt_path}")
        
    except Exception as e:
        logging.error(f">>> AURA PROFILER: Error generating summary: {e}")


def _clean_kernel_name(name: str) -> str:
    """
    Simplifies C++ mangled names (starting with _Z) into readable categories 
    to save tokens, while keeping Python paths and meaningful names intact.
    """
    if not name or not isinstance(name, str):
        return "unknown"
        
    # Pass through Python files, aten:: ops, and standard names
    if not name.startswith("_Z"):
        # Optional: Clean up standard wrapper noise if needed
        if "pybind11" in name: return "[pybind11_wrapper]"
        return name

    # Heuristic mapping for mangled CUDA kernels
    lower_name = name.lower()
    if "gemm" in lower_name: return "[GEMM Kernel]"
    if "convolution" in lower_name or "conv" in lower_name: return "[Conv Kernel]"
    if "elementwise" in lower_name: return "[Elementwise Kernel]"
    if "layer_norm" in lower_name: return "[LayerNorm Kernel]"
    if "softmax" in lower_name: return "[Softmax Kernel]"
    if "copy" in lower_name: return "[Copy Kernel]"
    if "add" in lower_name: return "[Add Kernel]"
    if "mul" in lower_name: return "[Mul Kernel]"
    if "activation" in lower_name or "silu" in lower_name or "relu" in lower_name: return "[Activation Kernel]"
    if "fft" in lower_name: return "[FFT Kernel]"
    if "scatter" in lower_name: return "[Scatter Kernel]"
    if "gather" in lower_name: return "[Gather Kernel]"
    if "index" in lower_name: return "[Indexing Kernel]"
    if "vectorized" in lower_name: return "[Vectorized Kernel]"
    
    return "[CUDA Kernel]"


def process_torch_trace_json(json_path: str, output_txt_path: str, top_k=50):
    """
    Reads a Torch Profiler Chrome Trace JSON.
    [FIXED] Sanitizes bad JSON, calculates Self Time, and aggregates SHAPES.
    """
    if not os.path.exists(json_path):
        return

    try:
        # 1. Read and Sanitize (Fix the """ bug)
        with open(json_path, 'r', encoding='utf-8') as f:
            raw_content = f.read()

        if '"""' in raw_content:
            logging.warning(f"Sanitizing corrupt Torch trace: {json_path}")
            raw_content = raw_content.replace('"""', "'''") 
            # Optional: Save back to disk so Chrome can open it
            # with open(json_path, 'w', encoding='utf-8') as f: f.write(raw_content)

        data = json.loads(raw_content)
    except Exception as e:
        logging.error(f"Error reading Torch trace {json_path}: {e}")
        return

    events = data.get('traceEvents', [])
    # Filter for duration events ('X') 
    events = [e for e in events if e.get('ph') == 'X' and 'dur' in e]
    
    if not events:
        return

    events.sort(key=lambda x: x['ts'])

    # Stats now tracks 'shapes' as a Counter
    stats = defaultdict(lambda: {
        'total': 0.0, 
        'self': 0.0, 
        'calls': 0, 
        'callers': Counter(),
        'shapes': Counter() 
    })
    
    stack = []

    for e in events:
        start = e['ts']
        dur = e['dur']
        end = start + dur
        name = e.get('name', 'unknown')
        
        # --- NEW: Extract Shapes ---
        # Torch usually stores shapes in args['Input Dims'] as a list of lists/ints
        # We stringify it to make it hashable for the Counter.
        args = e.get('args', {})
        shape_str = ""
        if 'Input Dims' in args:
            # Format: [[32, 64], [64, 128]]
            shape_info = args['Input Dims']
            # Make it compact: remove spaces
            shape_str = str(shape_info).replace(" ", "")
        
        # Update Stats
        stats[name]['total'] += dur
        stats[name]['calls'] += 1
        stats[name]['self'] += dur 
        if shape_str:
            stats[name]['shapes'][shape_str] += 1

        # Stack Logic for Self Time
        while stack and stack[-1]['end'] <= start:
            stack.pop()

        if stack:
            parent = stack[-1]
            stats[parent['name']]['self'] -= dur
            if stats[parent['name']]['self'] < 0: stats[parent['name']]['self'] = 0
            stats[name]['callers'][parent['name']] += dur

        stack.append({'name': name, 'end': end})

    # Prepare Results
    results = []
    for name, data in stats.items():
        # Get most common shape
        most_common_shape = ""
        if data['shapes']:
            # Grab the shape that appears most frequently
            most_common_shape, _ = data['shapes'].most_common(1)[0]
            # Truncate if insanely long (rare, but good for safety)
            if len(most_common_shape) > 40:
                most_common_shape = most_common_shape[:37] + "..."

        results.append({
            'name': _clean_kernel_name(name),
            'self_ms': data['self'] / 1000.0,
            'total_ms': data['total'] / 1000.0,
            'calls': data['calls'],
            'top_callers': data['callers'].most_common(3),
            'shape': most_common_shape
        })

    results.sort(key=lambda x: x['self_ms'], reverse=True)

    with open(output_txt_path, 'w', encoding='utf-8') as f:
        f.write(f"--- AURA TORCH KERNEL SUMMARY ---\n")
        f.write(f"Source: {json_path}\n")
        # Added INPUT SHAPES column
        f.write(f"FUNCTION/KERNEL | SELF (ms) | TOTAL (ms) | CALLS | INPUT SHAPES (Top)\n")
        f.write("---\n")
        
        for r in results[:top_k]:
            # Compact printing: Just the shape string, no labels
            shape_display = r['shape'] if r['shape'] else "-"
            f.write(f"{r['name']} | {r['self_ms']:.2f} | {r['total_ms']:.2f} | {r['calls']} | {shape_display}\n")
            
            if not r['top_callers']:
                continue
            for caller_name, caller_dur_us in r['top_callers']:
                caller_ms = caller_dur_us / 1000.0
                pct = (caller_ms / r['total_ms']) * 100 if r['total_ms'] > 0 else 0
                clean_caller = _clean_kernel_name(caller_name)
                clean_caller = clean_caller.replace("enumerate(DataLoader)#_SingleProcessDataLoaderIter.__next__", "DataLoader")
                f.write(f" {clean_caller} ({caller_ms:.1f}ms - {pct:.0f}%)\n")

    logging.info(f">>> AURA PROFILER: Torch Summary generated at {output_txt_path}")