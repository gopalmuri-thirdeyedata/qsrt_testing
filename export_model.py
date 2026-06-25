import os
import shutil
import ultralytics.utils.checks

# Monkeypatch check_requirements to bypass the strict onnx version constraint (<1.18.0)
# which fails on Python 3.13 because it attempts to compile onnx 1.17.0 from source.
original_check = ultralytics.utils.checks.check_requirements
def dummy_check(requirements, exclude=(), install=True, cmds=''):
    if isinstance(requirements, (list, tuple)):
        new_reqs = [r for r in requirements if 'onnx' not in str(r)]
        if len(new_reqs) < len(requirements):
            print(f"[Patch] Filtered out ONNX check from requirements list: {requirements}")
            requirements = new_reqs
            if not requirements:
                return
    elif isinstance(requirements, str):
        if 'onnx' in requirements:
            print(f"[Patch] Bypassing ONNX check for requirement: {requirements}")
            return
    return original_check(requirements, exclude, install, cmds)

ultralytics.utils.checks.check_requirements = dummy_check

from ultralytics import YOLO

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "best_16062026.pt")
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
    
    # Load PyTorch model
    print("Loading PyTorch model...")
    model = YOLO(model_path)
    
    # 1. Export 640x640 ONNX
    print("\n--- Exporting to ONNX (640x640) ---")
    onnx_path_640 = model.export(format="onnx", imgsz=640, opset=12)
    print(f"Exported to: {onnx_path_640}")
    
    # Rename default output to distinct name
    dest_640 = os.path.join(script_dir, "best_16062026_640.onnx")
    if os.path.exists(onnx_path_640):
        if os.path.exists(dest_640):        
            os.remove(dest_640)
        shutil.move(onnx_path_640, dest_640)
        print(f"Renamed to: {dest_640}")
        
    # 2. Export 320x320 ONNX
    print("\n--- Exporting to ONNX (320x320) ---")
    onnx_path_320 = model.export(format="onnx", imgsz=320, opset=12)
    print(f"Exported to: {onnx_path_320}")
    
    # Rename default output to distinct name
    dest_320 = os.path.join(script_dir, "best_03062026_320.onnx")
    if os.path.exists(onnx_path_320):
        if os.path.exists(dest_320):
            os.remove(dest_320)
        shutil.move(onnx_path_320, dest_320)
        print(f"Renamed to: {dest_320}")
        
    # 3. Export 640x640 OpenVINO
    print("\n--- Exporting to OpenVINO (640x640) ---")
    openvino_path_640 = model.export(format="openvino", imgsz=640)
    print(f"Exported to: {openvino_path_640}")
    
    # Rename default output folder to end with _openvino_model
    dest_ov_640 = os.path.join(script_dir, "best_03062026_640_openvino_model")
    if os.path.exists(openvino_path_640):
        if os.path.exists(dest_ov_640):
            shutil.rmtree(dest_ov_640)
        shutil.move(openvino_path_640, dest_ov_640)
        print(f"Renamed to: {dest_ov_640}")
        
    # 4. Export 320x320 OpenVINO
    print("\n--- Exporting to OpenVINO (320x320) ---")
    openvino_path_320 = model.export(format="openvino", imgsz=320)
    print(f"Exported to: {openvino_path_320}")
    
    # Rename default output folder to end with _openvino_model
    dest_ov_320 = os.path.join(script_dir, "best_03062026_320_openvino_model")
    if os.path.exists(openvino_path_320):
        if os.path.exists(dest_ov_320):
            shutil.rmtree(dest_ov_320)
        shutil.move(openvino_path_320, dest_ov_320)
        print(f"Renamed to: {dest_ov_320}")
        
    print("\nModel exports completed successfully!")

if __name__ == "__main__":
    main()
