import os
import io
import uuid
import base64
import time
import cv2
import numpy as np
import psutil
import threading
import ultralytics.utils.checks
from flask import Flask, request, jsonify, render_template, Response, send_from_directory
from werkzeug.utils import secure_filename

# ── GPU / CPU auto-detection ──────────────────────────────────────────────────
try:
    import torch
    if torch.cuda.is_available():
        DEVICE      = "cuda:0"
        GPU_NAME    = torch.cuda.get_device_name(0)
        GPU_MEM_GB  = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
        USE_GPU     = True
    else:
        DEVICE      = "cpu"
        GPU_NAME    = None
        GPU_MEM_GB  = None
        USE_GPU     = False
except ImportError:
    DEVICE     = "cpu"
    GPU_NAME   = None
    GPU_MEM_GB = None
    USE_GPU    = False

print(f"[DEVICE] Using: {DEVICE}" + (f" — {GPU_NAME} ({GPU_MEM_GB} GB)" if USE_GPU else " (no CUDA GPU found)"), flush=True)
# ─────────────────────────────────────────────────────────────────────────────

# ── Monkeypatch: bypass strict ONNX version check (Python 3.13) ──────────────
original_check = ultralytics.utils.checks.check_requirements
def dummy_check(requirements, exclude=(), install=True, cmds=''):
    if isinstance(requirements, (list, tuple)):
        new_reqs = [r for r in requirements if 'onnx' not in str(r)]
        if len(new_reqs) < len(requirements):
            requirements = new_reqs
            if not requirements:
                return
    elif isinstance(requirements, str):
        if 'onnx' in requirements:
            return
    return original_check(requirements, exclude, install, cmds)
ultralytics.utils.checks.check_requirements = dummy_check
# ─────────────────────────────────────────────────────────────────────────────

from ultralytics import YOLO

app = Flask(__name__)

@app.after_request
def add_cors_headers(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

# ── Directories ───────────────────────────────────────────────────────────────
SCRIPT_DIR      = os.path.dirname(os.path.abspath(__file__))
PT_MODEL_PATH   = os.path.join(SCRIPT_DIR, "best_16062026.pt")
UPLOAD_FOLDER   = os.path.join(SCRIPT_DIR, "uploads")
OUTPUT_FOLDER   = os.path.join(SCRIPT_DIR, "output_annotated")
INPUT_FOLDER    = os.path.join(SCRIPT_DIR, "input_videos")

for d in [UPLOAD_FOLDER, OUTPUT_FOLDER, INPUT_FOLDER]:
    os.makedirs(d, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ── Thread safety ────────────────────────────────────────────────────────────
# On CPU: lock prevents concurrent inference (avoids ntdll.dll crashes).
# On GPU: CUDA is thread-safe; lock is a no-op context manager.
class _NoLock:
    """Dummy context manager — used when GPU inference needs no serialisation."""
    def __enter__(self): return self
    def __exit__(self, *_): pass

inference_lock = threading.Lock() if not USE_GPU else _NoLock()

# ── Load PT model instances ───────────────────────────────────────────────────
print("Loading PyTorch model for image detection…", flush=True)
img_model = YOLO(PT_MODEL_PATH)

print("Loading PyTorch model for Video Feed 1…", flush=True)
feed1_model = YOLO(PT_MODEL_PATH)

print("Loading PyTorch model for Video Feed 2…", flush=True)
feed2_model = YOLO(PT_MODEL_PATH)

print("Loading PyTorch model for Batch processing…", flush=True)
batch_model = YOLO(PT_MODEL_PATH)

print("All models pre-loaded successfully!", flush=True)
print(f"[DEVICE] Inference device: {DEVICE}" + (f" ({GPU_NAME})" if USE_GPU else ""), flush=True)

# ── Backend log buffer ────────────────────────────────────────────────────────
device_label = f"GPU — {GPU_NAME} ({GPU_MEM_GB} GB)" if USE_GPU else "CPU (no CUDA GPU found)"
backend_logs      = [f"Backend started. Device: {device_label}. Model: PyTorch (.pt)"]
backend_logs_lock = threading.Lock()

def log_backend(msg):
    print(msg, flush=True)
    with backend_logs_lock:
        backend_logs.append(msg)

# ── CPU monitor ───────────────────────────────────────────────────────────────
try:
    _init_cores = psutil.cpu_percent(interval=0.1, percpu=True)
except Exception:
    _init_cores = [0.0] * (psutil.cpu_count() or 4)

cpu_stats_cache = {
    "cpu_percent": sum(_init_cores) / len(_init_cores) if _init_cores else 0.0,
    "cpu_cores": _init_cores
}

def _cpu_monitor():
    while True:
        try:
            cores = psutil.cpu_percent(interval=0.5, percpu=True)
            if cores:
                cpu_stats_cache["cpu_cores"]   = cores
                cpu_stats_cache["cpu_percent"] = sum(cores) / len(cores)
        except Exception:
            pass
        time.sleep(0.2)

threading.Thread(target=_cpu_monitor, daemon=True).start()

# ── Batch job store ───────────────────────────────────────────────────────────
# { job_id: { status, progress, frames_done, total_frames, fps, output_file, error } }
batch_jobs      = {}
batch_jobs_lock = threading.Lock()

# ── Batch live frame cache (latest annotated JPEG per job for preview) ────────
batch_frame_cache = {}   # { job_id: bytes (JPEG) }
batch_frame_lock  = threading.Lock()

# ──────────────────────────────────────────────────────────────────────────────
# ROUTES
# ──────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/get_logs")
def get_logs():
    with backend_logs_lock:
        logs = list(backend_logs)
        backend_logs.clear()
    return jsonify({"logs": logs})

@app.route("/log_config", methods=["POST"])
def log_config():
    data  = request.get_json() or {}
    param = data.get("parameter", "Unknown")
    value = data.get("value", "Unknown")
    log_backend(f"[UI CONFIG] {param} → {value}")
    return jsonify({"success": True})

@app.route("/cpu_stats")
def cpu_stats():
    return jsonify({
        "cpu_percent": cpu_stats_cache["cpu_percent"],
        "cpu_cores":   cpu_stats_cache["cpu_cores"],
        "core_count":  len(cpu_stats_cache["cpu_cores"])
    })

@app.route("/device_info")
def device_info():
    """Returns active inference device and GPU details (if available)."""
    info = {
        "device":    DEVICE,
        "use_gpu":   USE_GPU,
        "gpu_name":  GPU_NAME,
        "gpu_mem_gb": GPU_MEM_GB,
    }
    if USE_GPU:
        try:
            import torch
            info["gpu_mem_used_gb"] = round(torch.cuda.memory_allocated(0) / 1e9, 2)
            info["gpu_mem_reserved_gb"] = round(torch.cuda.memory_reserved(0) / 1e9, 2)
        except Exception:
            pass
    return jsonify(info)

# ── Image detection ───────────────────────────────────────────────────────────
@app.route("/detect", methods=["POST"])
def detect():
    if "image" not in request.files:
        return jsonify({"success": False, "error": "No image file provided"}), 400
    file     = request.files["image"]
    img_bytes = file.read()
    nparr    = np.frombuffer(img_bytes, np.uint8)
    img      = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"success": False, "error": "Invalid image format"}), 400

    img_size       = int(request.form.get("img_size", 640))
    conf_threshold = float(request.form.get("conf", 0.25))
    log_backend(f"[IMAGE] Detecting | imgsz={img_size} conf={conf_threshold}")

    start = time.time()
    with inference_lock:
        results = img_model(img, imgsz=img_size, conf=conf_threshold, device=DEVICE, verbose=False)
    inf_ms = (time.time() - start) * 1000
    result = results[0]

    annotated = result.plot()
    _, buf    = cv2.imencode('.png', annotated)
    b64       = base64.b64encode(buf).decode('utf-8')

    detections = []
    if result.boxes is not None:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
            detections.append({
                "class_name": img_model.names[int(box.cls[0])],
                "confidence": float(box.conf[0]),
                "bbox": [round(x1), round(y1), round(x2), round(y2)]
            })
    log_backend(f"[IMAGE] Done in {inf_ms:.1f}ms | {len(detections)} objects")
    return jsonify({"success": True, "image_base64": b64,
                    "inference_time_ms": round(inf_ms, 2),
                    "detections": detections, "total_detections": len(detections)})

# ── Video upload (dual-feed streaming) ───────────────────────────────────────
@app.route("/upload_video", methods=["POST"])
def upload_video():
    if "video" not in request.files:
        return jsonify({"success": False, "error": "No video file provided"}), 400
    file     = request.files["video"]
    filename = secure_filename(file.filename)
    filename = f"{int(time.time())}_{filename}"
    path     = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(path)
    log_backend(f"[VIDEO UPLOAD] {file.filename} → {filename}")
    return jsonify({"success": True, "filename": filename})

def _draw_cached(frame, last_res, model):
    if last_res is None or last_res.boxes is None:
        return frame
    annotated = frame.copy()
    boxes = last_res.boxes
    if len(boxes) == 0:
        return annotated
    xyxy  = boxes.xyxy.cpu().numpy()
    ids   = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else None
    confs = boxes.conf.cpu().numpy()
    clss  = boxes.cls.cpu().numpy().astype(int)
    for i in range(len(xyxy)):
        x1, y1, x2, y2 = map(int, xyxy[i])
        label = model.names[clss[i]]
        conf  = confs[i]
        cid   = ids[i] if ids is not None else clss[i]
        h     = hash(str(cid))
        color = ((h & 0x7F)+64, ((h>>8)&0x7F)+64, ((h>>16)&0x7F)+64)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        text  = f"{label}{'#'+str(ids[i]) if ids is not None else ''} {conf:.2f}"
        (w, ht), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        yb    = max(y1-18, 0)
        cv2.rectangle(annotated, (x1, yb), (x1+w, yb+18), color, -1)
        cv2.putText(annotated, text, (x1, yb+13), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)
    return annotated

def gen_frames(video_path, imgsz, conf, target_fps, feed_id):
    cap = cv2.VideoCapture(video_path)
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    if orig_fps <= 0 or np.isnan(orig_fps):
        orig_fps = 30.0
    orig_fps = max(5.0, min(orig_fps, 60.0))
    frame_delay    = 1.0 / orig_fps
    frame_interval = max(1, int(round(orig_fps / target_fps)))
    model = feed1_model if feed_id == 1 else feed2_model
    log_backend(f"[STREAM Feed{feed_id}] Start | {os.path.basename(video_path)} imgsz={imgsz} fps={target_fps}")
    frame_idx, last_res = 0, None
    try:
        while cap.isOpened():
            t0  = time.time()
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_interval == 0:
                with inference_lock:
                    results = model.track(frame, persist=True, tracker="bytetrack.yaml",
                                          imgsz=imgsz, conf=conf, device=DEVICE, verbose=False)
                last_res = results[0]
                annotated = last_res.plot()
                n = len(last_res.boxes) if last_res.boxes is not None else 0
                log_backend(f"[STREAM Feed{feed_id}] Frame {frame_idx:04d} | {(time.time()-t0)*1000:.0f}ms | {n} det")
            else:
                annotated = _draw_cached(frame, last_res, model)
            ret2, jpeg = cv2.imencode('.jpg', annotated)
            if not ret2:
                continue
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
            frame_idx += 1
            elapsed = time.time() - t0
            if frame_delay - elapsed > 0:
                time.sleep(frame_delay - elapsed)
    except Exception as e:
        log_backend(f"[STREAM Feed{feed_id}] Error: {e}")
    finally:
        cap.release()
        log_backend(f"[STREAM Feed{feed_id}] Closed.")

@app.route("/video_feed/<filename>")
def video_feed(filename):
    path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    if not os.path.exists(path):
        return "Video not found", 404
    imgsz    = int(request.args.get("img_size", 320))
    conf     = float(request.args.get("conf", 0.25))
    fps      = float(request.args.get("target_fps", 5.0))
    feed_id  = int(request.args.get("feed_id", 1))
    return Response(gen_frames(path, imgsz, conf, fps, feed_id),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

# ──────────────────────────────────────────────────────────────────────────────
# BATCH PROCESSING ROUTES
# ──────────────────────────────────────────────────────────────────────────────

@app.route("/batch_upload", methods=["POST"])
def batch_upload():
    """Upload a video for batch annotation."""
    if "video" not in request.files:
        return jsonify({"success": False, "error": "No video file"}), 400
    file     = request.files["video"]
    filename = secure_filename(file.filename)
    filename = f"batch_{int(time.time()*1000)}_{filename}"
    path     = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(path)
    log_backend(f"[BATCH UPLOAD] {file.filename} → {filename}")
    return jsonify({"success": True, "filename": filename})


def _run_batch_job(job_id, video_path, imgsz, conf):
    """Background thread: annotate video and save to output_annotated/."""
    with batch_jobs_lock:
        batch_jobs[job_id]["status"] = "processing"

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        with batch_jobs_lock:
            batch_jobs[job_id].update({"status": "error", "error": "Cannot open video"})
        return

    fps      = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    with batch_jobs_lock:
        batch_jobs[job_id]["total_frames"] = total

    stem      = os.path.splitext(os.path.basename(video_path))[0]
    out_name  = f"{stem}_annotated.mp4"
    out_path  = os.path.join(OUTPUT_FOLDER, out_name)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    log_backend(f"[BATCH JOB {job_id[:8]}] Processing: {os.path.basename(video_path)} | {width}x{height} @ {fps:.1f}fps | {total} frames")

    frame_idx = 0
    t_start   = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            with inference_lock:
                results = batch_model.track(
                    frame, persist=True, tracker="bytetrack.yaml",
                    imgsz=imgsz, conf=conf, device=DEVICE, verbose=False
                )
            annotated = results[0].plot()
            writer.write(annotated)
            frame_idx += 1

            # Cache latest frame as JPEG for live UI preview (every frame)
            _, jpeg = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 72])
            with batch_frame_lock:
                batch_frame_cache[job_id] = jpeg.tobytes()

            # Update progress every 10 frames
            if frame_idx % 10 == 0:
                elapsed  = time.time() - t_start
                cur_fps  = frame_idx / elapsed if elapsed > 0 else 0
                progress = round((frame_idx / total * 100), 1) if total > 0 else 0
                with batch_jobs_lock:
                    batch_jobs[job_id].update({
                        "frames_done": frame_idx,
                        "progress":    progress,
                        "fps":         round(cur_fps, 2)
                    })
                if frame_idx % 100 == 0:
                    log_backend(f"[BATCH JOB {job_id[:8]}] Frame {frame_idx}/{total} ({progress}%) @ {cur_fps:.1f}fps")

    except Exception as e:
        log_backend(f"[BATCH JOB {job_id[:8]}] Error: {e}")
        with batch_jobs_lock:
            batch_jobs[job_id].update({"status": "error", "error": str(e)})
        cap.release()
        writer.release()
        # Clear frame cache on error
        with batch_frame_lock:
            batch_frame_cache.pop(job_id, None)
        return
    finally:
        cap.release()
        writer.release()

    elapsed = time.time() - t_start
    log_backend(f"[BATCH JOB {job_id[:8]}] Done in {elapsed:.1f}s → {out_name}")

    # Clear frame cache after job finishes
    with batch_frame_lock:
        batch_frame_cache.pop(job_id, None)

    with batch_jobs_lock:
        batch_jobs[job_id].update({
            "status":      "done",
            "progress":    100,
            "frames_done": frame_idx,
            "fps":         round(frame_idx / elapsed, 2) if elapsed > 0 else 0,
            "output_file": out_name
        })


@app.route("/batch_process", methods=["POST"])
def batch_process():
    """Start a batch annotation job in a background thread."""
    data     = request.get_json() or {}
    filename = data.get("filename")
    if not filename:
        return jsonify({"success": False, "error": "No filename provided"}), 400

    video_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    if not os.path.exists(video_path):
        return jsonify({"success": False, "error": "Video file not found"}), 404

    imgsz = int(data.get("imgsz", 640))
    conf  = float(data.get("conf", 0.25))
    job_id = str(uuid.uuid4())

    with batch_jobs_lock:
        batch_jobs[job_id] = {
            "status":      "queued",
            "progress":    0,
            "frames_done": 0,
            "total_frames": 0,
            "fps":         0,
            "output_file": None,
            "error":       None
        }

    t = threading.Thread(target=_run_batch_job, args=(job_id, video_path, imgsz, conf), daemon=True)
    t.start()
    log_backend(f"[BATCH] Job {job_id[:8]} queued for {filename}")
    return jsonify({"success": True, "job_id": job_id})


@app.route("/batch_status/<job_id>")
def batch_status(job_id):
    """Poll the status of a batch job."""
    with batch_jobs_lock:
        job = batch_jobs.get(job_id)
    if job is None:
        return jsonify({"success": False, "error": "Unknown job ID"}), 404
    return jsonify({"success": True, **job})



@app.route("/batch_preview/<job_id>")
def batch_preview(job_id):
    """Live MJPEG stream of the latest annotated frame from an active batch job."""
    def _generate():
        while True:
            with batch_jobs_lock:
                job = batch_jobs.get(job_id)
            if job is None:
                break
            with batch_frame_lock:
                frame = batch_frame_cache.get(job_id)
            if frame:
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            if job["status"] in ("done", "error"):
                break
            time.sleep(0.08)

    return Response(_generate(), mimetype="multipart/x-mixed-replace; boundary=frame")
@app.route("/download_annotated/<filename>")
def download_annotated(filename):
    """Serve a processed annotated video for download."""
    safe = secure_filename(filename)
    path = os.path.join(OUTPUT_FOLDER, safe)
    if not os.path.exists(path):
        return "File not found", 404
    return send_from_directory(OUTPUT_FOLDER, safe, as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
