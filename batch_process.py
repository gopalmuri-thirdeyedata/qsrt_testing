import os
import time
import cv2
import ultralytics.utils.checks
from ultralytics import YOLO

# ── Monkeypatch: bypass strict ONNX version check (Python 3.13 compat) ──────
_original_check = ultralytics.utils.checks.check_requirements
def _dummy_check(requirements, exclude=(), install=True, cmds=""):
    if isinstance(requirements, (list, tuple)):
        requirements = [r for r in requirements if "onnx" not in str(r)]
        if not requirements:
            return
    elif isinstance(requirements, str) and "onnx" in requirements:
        return
    return _original_check(requirements, exclude, install, cmds)
ultralytics.utils.checks.check_requirements = _dummy_check
# ────────────────────────────────────────────────────────────────────────────

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
PT_MODEL_PATH  = os.path.join(SCRIPT_DIR, "best_16062026.pt")
INPUT_DIR      = os.path.join(SCRIPT_DIR, "input_videos")
OUTPUT_DIR     = os.path.join(SCRIPT_DIR, "output_annotated")

SUPPORTED_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}

# ── Inference settings ───────────────────────────────────────────────────────
IMGSZ          = 640      # inference resolution
CONF_THRESHOLD = 0.25     # minimum confidence to show a detection
DEVICE         = "cpu"    # change to "cuda:0" if you have a GPU
USE_TRACKING   = True     # True -> ByteTrack tracking, False -> plain detect
# ────────────────────────────────────────────────────────────────────────────


def process_video(model: YOLO, video_path: str, output_path: str) -> None:
    """Run YOLOv8 inference on every frame and write annotated video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [ERROR] Cannot open video: {video_path}")
        return

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"  Resolution : {width}x{height}  |  FPS: {fps:.1f}  |  Frames: {total}")

    frame_idx   = 0
    start_time  = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if USE_TRACKING:
            results = model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                imgsz=IMGSZ,
                conf=CONF_THRESHOLD,
                device=DEVICE,
                verbose=False,
            )
        else:
            results = model(
                frame,
                imgsz=IMGSZ,
                conf=CONF_THRESHOLD,
                device=DEVICE,
                verbose=False,
            )

        annotated = results[0].plot()
        writer.write(annotated)
        frame_idx += 1

        # Progress log every 50 frames
        if frame_idx % 50 == 0:
            elapsed  = time.time() - start_time
            fps_real = frame_idx / elapsed if elapsed > 0 else 0
            pct      = (frame_idx / total * 100) if total > 0 else 0
            print(f"  Frame {frame_idx}/{total} ({pct:.1f}%)  |  {fps_real:.1f} fps")

    cap.release()
    writer.release()

    elapsed = time.time() - start_time
    print(f"  Done in {elapsed:.1f}s  ->  Saved: {output_path}")


def main():
    # ── Validate paths ───────────────────────────────────────────────────────
    if not os.path.exists(PT_MODEL_PATH):
        print(f"[ERROR] PyTorch model not found: {PT_MODEL_PATH}")
        return

    os.makedirs(INPUT_DIR,  exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Gather input videos ──────────────────────────────────────────────────
    video_files = sorted([
        f for f in os.listdir(INPUT_DIR)
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXTS
    ])

    if not video_files:
        print(f"[INFO] No videos found in: {INPUT_DIR}")
        print(f"       Drop your videos there (mp4/avi/mov/mkv/wmv/flv/webm) and re-run.")
        return

    print(f"\n{'='*60}")
    print(f"  ThirdEye Batch Annotator  --  PyTorch (.pt) model only")
    print(f"{'='*60}")
    print(f"  Model   : {PT_MODEL_PATH}")
    print(f"  Input   : {INPUT_DIR}  ({len(video_files)} video(s))")
    print(f"  Output  : {OUTPUT_DIR}")
    print(f"  Imgsz   : {IMGSZ}  |  Conf: {CONF_THRESHOLD}  |  Device: {DEVICE}")
    print(f"  Tracking: {'ByteTrack ON' if USE_TRACKING else 'Detect-only'}")
    print(f"{'='*60}\n")

    # ── Load model once ──────────────────────────────────────────────────────
    print("Loading PyTorch model ...")
    model = YOLO(PT_MODEL_PATH)
    print("Model loaded.\n")

    # ── Process each video ───────────────────────────────────────────────────
    overall_start = time.time()

    for idx, filename in enumerate(video_files, start=1):
        input_path  = os.path.join(INPUT_DIR, filename)
        stem, _     = os.path.splitext(filename)
        output_name = f"{stem}_annotated.mp4"
        output_path = os.path.join(OUTPUT_DIR, output_name)

        print(f"[{idx}/{len(video_files)}] Processing: {filename}")
        process_video(model, input_path, output_path)
        print()

    total_elapsed = time.time() - overall_start
    print(f"{'='*60}")
    print(f"  All {len(video_files)} video(s) processed in {total_elapsed:.1f}s")
    print(f"  Annotated videos saved to: {OUTPUT_DIR}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
