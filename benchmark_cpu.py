import os
import time
import cv2
import numpy as np
from ultralytics import YOLO

def benchmark_model(model_name, model_path, video_path, limit_frames=100, imgsz=640):
    print(f"\n==========================================")
    print(f"Benchmarking: {model_name}")
    print(f"Model Path: {model_path}")
    print(f"Resolution: {imgsz}x{imgsz}")
    print(f"==========================================")
    
    # 1. Measure model load time
    start_load = time.perf_counter()
    try:
        model = YOLO(model_path, task="detect")
    except Exception as e:
        print(f"Failed to load model {model_name}: {e}")
        return None
    load_time = time.perf_counter() - start_load
    print(f"Model Load Time: {load_time:.3f} seconds")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_video = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video Info: {total_frames} total frames @ {fps_video:.2f} FPS")
    
    frames_to_process = min(limit_frames, total_frames) if limit_frames > 0 else total_frames
    print(f"Processing first {frames_to_process} frames...")
    
    inference_times = []
    e2e_times = []
    total_detections = 0
    confidences = []
    
    frame_idx = 0
    while cap.isOpened() and frame_idx < frames_to_process:
        start_e2e = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            break
        
        # Measure only raw inference time
        start_inf = time.perf_counter()
        # Explicitly set device="cpu" and the target imgsz
        results = model(frame, imgsz=imgsz, device="cpu", verbose=False, conf=0.1)
        inf_time = time.perf_counter() - start_inf
        inference_times.append(inf_time)
        
        # Record detection stats
        result = results[0]
        boxes = result.boxes
        if boxes is not None:
            num_det = len(boxes)
            total_detections += num_det
            if num_det > 0:
                confidences.extend(boxes.conf.cpu().numpy().tolist())
                
        e2e_time = time.perf_counter() - start_e2e
        e2e_times.append(e2e_time)
        frame_idx += 1
        
        if frame_idx % 20 == 0:
            print(f"Processed {frame_idx}/{frames_to_process} frames...")
            
    cap.release()
    
    if len(inference_times) == 0:
        print("No frames processed.")
        return None
        
    avg_inf_time = np.mean(inference_times)
    p50_inf_time = np.percentile(inference_times, 50)
    p95_inf_time = np.percentile(inference_times, 95)
    avg_fps = 1.0 / avg_inf_time
    
    avg_e2e_time = np.mean(e2e_times)
    e2e_fps = 1.0 / avg_e2e_time
    
    avg_conf = np.mean(confidences) if len(confidences) > 0 else 0.0
    
    print(f"\n--- Results for {model_name} ---")
    print(f"Avg Inference Time: {avg_inf_time*1000:.2f} ms per frame")
    print(f"50th Percentile (p50) Inference Time: {p50_inf_time*1000:.2f} ms")
    print(f"95th Percentile (p95) Inference Time: {p95_inf_time*1000:.2f} ms")
    print(f"Inference-only FPS: {avg_fps:.2f}")
    print(f"End-to-End FPS (incl. decoding): {e2e_fps:.2f}")
    print(f"Total Detections: {total_detections}")
    print(f"Avg Confidence: {avg_conf:.4f}")
    
    return {
        "model_name": model_name,
        "load_time": load_time,
        "avg_inf_time_ms": avg_inf_time * 1000,
        "p50_inf_time_ms": p50_inf_time * 1000,
        "p95_inf_time_ms": p95_inf_time * 1000,
        "inf_fps": avg_fps,
        "e2e_fps": e2e_fps,
        "total_detections": total_detections,
        "avg_confidence": avg_conf
    }

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    video_filename = 'djdas@thirdeyedata.ai_20260601_103019-110751_8691554261_ch3_3.mp4'
    video_path = os.path.join(script_dir, video_filename)
    
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return
        
    configs = [
        {
            "name": "PyTorch Baseline (640x640)",
            "path": os.path.join(script_dir, "best_03062026.pt"),
            "imgsz": 640
        },
        {
            "name": "ONNX (640x640)",
            "path": os.path.join(script_dir, "best_03062026_640.onnx"),
            "imgsz": 640
        },
        {
            "name": "ONNX (320x320)",
            "path": os.path.join(script_dir, "best_03062026_320.onnx"),
            "imgsz": 320
        },
        {
            "name": "OpenVINO (640x640)",
            "path": os.path.join(script_dir, "best_03062026_640_openvino_model"),
            "imgsz": 640
        },
        {
            "name": "OpenVINO (320x320)",
            "path": os.path.join(script_dir, "best_03062026_320_openvino_model"),
            "imgsz": 320
        }
    ]
    
    results = []
    for cfg in configs:
        if os.path.exists(cfg["path"]):
            res = benchmark_model(cfg["name"], cfg["path"], video_path, limit_frames=100, imgsz=cfg["imgsz"])
            if res:
                results.append(res)
        else:
            print(f"\nSkipping {cfg['name']} (File not found: {cfg['path']})")
            
    if not results:
        print("No benchmarks succeeded.")
        return
        
    # Print comparison table
    print("\n" + "="*80)
    print("                      YOLO CPU BENCHMARK COMPARISON")
    print("="*80)
    print(f"{'Model Format & Resolution':<30} | {'Load (s)':<8} | {'Avg Inf (ms)':<12} | {'Inf FPS':<8} | {'E2E FPS':<8} | {'Detections':<10} | {'Avg Conf':<8}")
    print("-"*80)
    for r in results:
        print(f"{r['model_name']:<30} | {r['load_time']:<8.2f} | {r['avg_inf_time_ms']:<12.2f} | {r['inf_fps']:<8.2f} | {r['e2e_fps']:<8.2f} | {r['total_detections']:<10} | {r['avg_confidence']:<8.4f}")
    print("="*80)

if __name__ == "__main__":
    main()
