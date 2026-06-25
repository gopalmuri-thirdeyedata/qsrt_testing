# ThirdEye — YOLO Live CPU/GPU Inference Dashboard

A real-time object detection and tracking dashboard built with **YOLOv8 (PyTorch .pt)**, **Flask**, and **OpenCV**.  
Supports single-image detection, dual-video live streaming, and **bulk batch video annotation** — with automatic GPU acceleration when a CUDA device is available.

---

## ✨ Features

| Feature | Details |
|---|---|
| 🖼️ **Image Detection** | Upload any image → instant YOLO inference → annotated result |
| 🎥 **Dual Video Streams** | Two simultaneous MJPEG streams with ByteTrack object tracking |
| ⚡ **Batch Video Processing** | Drag & drop multiple videos → annotated MP4s saved to `output_annotated/` |
| 🟢 **GPU Auto-Detection** | Uses `cuda:0` if available, falls back to CPU automatically |
| 📊 **Live CPU Monitor** | Per-core utilisation bars updated every 500 ms |
| 🔋 **Live VRAM Monitor** | GPU VRAM usage bar (only shown when GPU is active) |
| 📋 **System Logs Console** | Real-time backend logs streamed to the browser |

---

## 🗂️ Project Structure

```
qsrt_testing/                    ← GitHub repo root
├── app.py                       # Flask backend — all routes & inference logic
├── batch_process.py             # Standalone CLI batch processor (no Flask needed)
├── export_model.py              # Export .pt model utility
├── benchmark_cpu.py             # CPU benchmark utility
├── best_16062026.pt             # YOLOv8 model weights  ← required (not in git)
├── .gitignore
├── README.md
├── templates/
│   └── index.html               # Single-page dashboard UI
├── input_videos/                # Drop videos here for CLI batch processing  (git-ignored)
├── output_annotated/            # Annotated outputs saved here               (git-ignored)
└── uploads/                     # Temporary upload storage (auto-created)    (git-ignored)
```

---

## ⚙️ Setup

### 1. Prerequisites

- Python **3.10 – 3.13**
- `pip` (or `conda`)
- *(Optional)* NVIDIA GPU with CUDA 11.8+ for GPU acceleration

---

### 2. Clone the repo

```bash
git clone https://github.com/gopalmuri-thirdeyedata/qsrt_testing.git
cd qsrt_testing
```

---

### 3. Create a virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

---

### 4. Install dependencies

```bash
pip install flask werkzeug opencv-python ultralytics psutil
```

**For GPU support (CUDA 11.8):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**For GPU support (CUDA 12.1):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**CPU only (no GPU):**
```bash
pip install torch torchvision
```

> **Python 3.13 note:** The ONNX version check is automatically patched at startup — no action needed.

---

### 5. Add your model

Place your YOLOv8 model weights in the project root:

```
qsrt_testing/
└── best_16062026.pt   ← put your model here
```

> The app will refuse to start if this file is missing.

---

### 6. Run the server

```bash
python app.py
```

Open your browser at: **http://127.0.0.1:5000**

The console will confirm which device is being used:
```
[DEVICE] Using: cuda:0 — NVIDIA RTX 4090 (24.0 GB)
```
or
```
[DEVICE] Using: cpu (no CUDA GPU found)
```

---

## 🚀 Usage

### Image Mode
1. Click **Image Mode** tab
2. Drag & drop or click to upload any image
3. YOLO detections appear instantly on the right

### Dual Video Mode
1. Click **Dual Video** tab
2. Upload one or two video files
3. Click **Start Parallel Analysis** to begin MJPEG streaming with tracking

### ⚡ Batch Process (bulk annotation)
1. Click **⚡ Batch Process** tab
2. Drag & drop **multiple videos** at once (or click to browse)
3. Click **▶ Start Processing**
4. Watch each video's **live progress bar** fill as frames are processed
5. When done, a **⬇ Download** link appears for each annotated video
6. All outputs are also saved to `output_annotated/`

### CLI Batch (no browser needed)
Drop videos into `input_videos/` then run:
```bash
python batch_process.py
```

---

## 🔧 Configuration

Edit the top of `batch_process.py` or use the UI sliders:

| Setting | Default | Description |
|---|---|---|
| `IMGSZ` | `640` | Inference image size |
| `CONF_THRESHOLD` | `0.25` | Minimum detection confidence |
| `DEVICE` | auto | `"cpu"` or `"cuda:0"` (auto-detected) |
| `USE_TRACKING` | `True` | ByteTrack object tracking |

---

## 📦 Requirements Summary

```
flask
werkzeug
opencv-python
ultralytics
psutil
torch
torchvision
```

---

## 📄 License

MIT — see [LICENSE](LICENSE) for details.
