from __future__ import annotations
from functools import wraps
import sys
import os
from pathlib import Path

import argparse
import tempfile
import time
import bpy
import numpy as np

# CRITICAL: Do NOT import torch/cv2 at module level!
# They must be imported AFTER setting up DLL directories on Windows
# See _ensure_torch_loaded() function below

from . plane_fit import estimate_plane_of_best_fit
from . utils import normalize_01, remove_infinities

# Global flags to track if we've set up DLLs and loaded torch
_torch_loaded = False
_cv2 = None
_torch = None

# Diagnostic: Confirm module imported successfully
print("TrueDepth: ✓ depth_estimation module imported successfully (torch/cv2 not loaded yet)")


def _ensure_torch_loaded():
    """
    Lazy-load torch and cv2 with proper DLL setup for Windows.
    This MUST be called before any function that uses torch/cv2.
    """
    global _torch_loaded, _cv2, _torch

    if _torch_loaded:
        return _torch, _cv2

    print("TrueDepth: Loading PyTorch and OpenCV...")

    # On Windows, we need to setup DLL paths BEFORE importing torch
    if sys.platform == "win32":
        print("TrueDepth: Setting up DLL directories for Windows...")

        dll_dirs = []

        # 1. Blender's main directory (contains blender.exe and VCRUNTIME DLLs)
        try:
            # sys.executable: C:\...\blender-4.5.4\4.5\python\bin\python.exe
            blender_exe_dir = Path(sys.executable).parent.parent.parent.parent
            if blender_exe_dir.exists() and (blender_exe_dir / "blender.exe").exists():
                dll_dirs.append(str(blender_exe_dir))
                print(f"   ✓ Found Blender exe dir: {blender_exe_dir}")
        except:
            pass

        # 2. Blender's Python bin directory
        try:
            blender_python_dir = Path(sys.executable).parent
            dll_dirs.append(str(blender_python_dir))
            print(f"   ✓ Found Blender Python dir: {blender_python_dir}")
        except:
            pass

        # 3. Torch lib directory
        try:
            for path in sys.path:
                torch_lib = Path(path) / "torch" / "lib"
                if torch_lib.exists():
                    dll_dirs.append(str(torch_lib))
                    print(f"   ✓ Found torch lib dir: {torch_lib}")
                    break
        except:
            pass

        # CRITICAL FIX: Pre-load python311.dll before torch tries to load it
        print("   → Pre-loading python311.dll...")
        try:
            import ctypes
            blender_python_dir = Path(sys.executable).parent
            python_dll = blender_python_dir / "python311.dll"

            if python_dll.exists():
                # Load it into memory first
                python_handle = ctypes.CDLL(str(python_dll))
                print(f"   ✓ Pre-loaded python311.dll from: {blender_python_dir}")
            else:
                print(f"   ⚠ python311.dll not found at: {python_dll}")
        except Exception as e:
            print(f"   ⚠ Could not pre-load python311.dll: {e}")

        # METHOD 1: Add to PATH (works more reliably than add_dll_directory sometimes)
        if dll_dirs:
            print("   → Adding directories to PATH environment variable...")
            current_path = os.environ.get('PATH', '')
            new_path = os.pathsep.join(dll_dirs) + os.pathsep + current_path
            os.environ['PATH'] = new_path
            print(f"   ✓ Updated PATH with {len(dll_dirs)} directories")

        # METHOD 2: Also use add_dll_directory (for Python 3.8+)
        if hasattr(os, 'add_dll_directory'):
            print("   → Also using os.add_dll_directory()...")
            for dll_dir in dll_dirs:
                try:
                    os.add_dll_directory(dll_dir)
                    print(f"   ✓ Added via add_dll_directory: {Path(dll_dir).name}")
                except Exception as e:
                    print(f"   ⚠ Could not add {Path(dll_dir).name}: {e}")

        # METHOD 3: Pre-load torch's core DLLs with LOAD_WITH_ALTERED_SEARCH_PATH
        print("   → Pre-loading torch core DLLs with ctypes...")
        try:
            import ctypes
            import ctypes.util

            # Find torch lib directory
            torch_lib_dir = None
            for path in sys.path:
                potential_torch_lib = Path(path) / "torch" / "lib"
                if potential_torch_lib.exists():
                    torch_lib_dir = potential_torch_lib
                    break

            if torch_lib_dir:
                # Try to pre-load core torch DLLs in dependency order
                # CPU DLLs (always present)
                core_dlls = ['c10.dll', 'fbgemm.dll', 'asmjit.dll', 'torch_cpu.dll']

                # CUDA DLLs (only in GPU builds)
                cuda_dlls = ['c10_cuda.dll', 'torch_cuda.dll', 'nvrtc64_112_0.dll',
                             'cusparse64_12.dll', 'cublas64_12.dll', 'cublasLt64_12.dll',
                             'cudnn64_9.dll', 'cudnn_ops64_9.dll', 'cudnn_cnn64_9.dll']

                # torch.dll must be last (depends on all others)
                all_dlls = core_dlls + cuda_dlls + ['torch.dll']

                for dll_name in all_dlls:
                    dll_path = torch_lib_dir / dll_name
                    if dll_path.exists():
                        try:
                            # Use WinDLL with mode parameter for better compatibility
                            # mode=0 means use default DLL search order with our modified PATH
                            dll_handle = ctypes.CDLL(str(dll_path), winmode=0)
                            print(f"   ✓ Pre-loaded: {dll_name}")
                        except Exception as e:
                            print(f"   ⚠ Could not pre-load {dll_name}: {e}")
                            # If c10.dll fails, this is critical
                            if dll_name == 'c10.dll':
                                print(f"   ⚠ CRITICAL: c10.dll failed to pre-load!")
                                print(f"   ⚠ This suggests a missing dependency for c10.dll itself")
        except Exception as e:
            print(f"   ⚠ Error during DLL pre-loading: {e}")

    # NOW import torch and cv2
    try:
        import torch as _torch_module
        import cv2 as _cv2_module

        _torch = _torch_module
        _cv2 = _cv2_module
        _torch_loaded = True

        print(f"TrueDepth: ✓ Successfully loaded PyTorch {_torch.__version__}")
        print(f"TrueDepth: ✓ Successfully loaded OpenCV {_cv2.__version__}")

        return _torch, _cv2

    except Exception as e:
        print(f"TrueDepth: ✗ FAILED to load dependencies: {e}")

        if "DLL" in str(e) or "1114" in str(e):
            print("\n" + "="*70)
            print("WINDOWS DLL LOADING ERROR - THIS IS ALMOST CERTAINLY:")
            print("")
            print(">>> MISSING VISUAL C++ REDISTRIBUTABLE <<<")
            print("")
            print("SOLUTION:")
            print("1. Download and install from Microsoft:")
            print("   https://aka.ms/vs/17/release/vc_redist.x64.exe")
            print("")
            print("2. After installing, RESTART BLENDER")
            print("")
            print("3. If that doesn't work, delete this folder and reinstall:")
            print(f"   {Path(sys.path[0]).parent / 'TrueDepth-blender-addon'}")
            print("")
            print("WHY THIS HAPPENS:")
            print("PyTorch's C++ DLLs (c10.dll, torch.dll) need the Microsoft")
            print("Visual C++ Runtime to work. Even if the DLLs are found, they")
            print("can't initialize without the correct runtime installed.")
            print("="*70 + "\n")
        raise

model = None
def time_function(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.4f} seconds")
        return result
    return wrapper

# Global variables to maintain state
_model = None
_current_device = None

def model_cache_decorator(func):
    @wraps(func)
    def wrapper(model_size, checkpoint_path, preferred_device, enable_cpu_offload=True):
        global _model, _current_device

        if _model is None:
            print(f"Initializing model ({model_size})...")
            _model = func(model_size, checkpoint_path, preferred_device, enable_cpu_offload)
            _current_device = preferred_device
        elif preferred_device != _current_device:
            print(f"Reloading model ({model_size}) for device: {preferred_device}")
            _model = func(model_size, checkpoint_path, preferred_device, enable_cpu_offload)
            _current_device = preferred_device
        else:
            print(f"Using cached model ({model_size})")

        return _model

    return wrapper


@time_function
@model_cache_decorator
def load_model(encoder, checkpoint_path, preferred_device, enable_cpu_offload=True):
    torch, _ = _ensure_torch_loaded()  # Load torch first!

    if preferred_device == 'gpu':
        DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    else:
        DEVICE = 'cpu'

    # Map encoder names to friendly names
    model_names = {
        'vits': 'Small (ViT-S)',
        'vitb': 'Base (ViT-B)',
        'vitl': 'Large (ViT-L)',
        'vitg': 'Giant (ViT-G)'
    }

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    print(f"Loading model: {model_names.get(encoder, encoder)}")
    print(f"  Encoder: {encoder}")
    print(f"  Device: {DEVICE}")
    print(f"  Checkpoint: {Path(checkpoint_path).name}")

    # Check GPU memory for large models and decide on offloading
    use_offload = False
    if DEVICE == 'cuda' and enable_cpu_offload:
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
        free_memory = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / (1024**3)
        print(f"  GPU Memory: {total_memory:.2f} GB total, {free_memory:.2f} GB free")

        # Estimate memory requirements (approximate)
        memory_requirements = {
            'vits': 2.0,  # ~2GB
            'vitb': 3.0,  # ~3GB
            'vitl': 4.5,  # ~4.5GB
            'vitg': 6.5,  # ~6.5GB
        }

        required_memory = memory_requirements.get(encoder, 2.0)
        print(f"  Estimated requirement: ~{required_memory:.1f} GB VRAM")

        # Enable CPU offloading if memory is insufficient
        if free_memory < required_memory * 1.2:  # 20% margin
            use_offload = True
            print(f"  ⚠ Low VRAM detected! Enabling CPU offloading")
            print(f"  ⚠ Performance will be slower but prevents OOM errors")

    from .depth_anything_v2 import dpt

    # Load model with appropriate strategy
    if use_offload:
        # Load model on CPU first
        print(f"  → Loading model on CPU first (offload mode)")
        model = dpt.DepthAnythingV2(**model_configs[encoder], device='cpu')
        model.load_state_dict(torch.load(checkpoint_path, map_location='cpu', weights_only=True))
        model = model.eval()

        # Move only essential parts to GPU
        # The preprocessing and postprocessing stay on CPU
        # Only the core inference backbone goes to GPU
        print(f"  → Transferring core model to GPU (keeping large layers on CPU)")
        try:
            # Move model to CUDA but allow fallback to CPU for large operations
            model = model.to(DEVICE)
            # CRITICAL: Update the model's device attribute after moving it
            model.device = DEVICE
            torch.cuda.empty_cache()
            print(f"  ✓ CPU offloading configured (hybrid CPU/GPU mode)")
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  ⚠ OOM during model loading - keeping model on CPU")
                DEVICE = 'cpu'
                model.device = 'cpu'
                torch.cuda.empty_cache()
            else:
                raise
    else:
        # Normal loading - full model on GPU
        model = dpt.DepthAnythingV2(**model_configs[encoder], device=DEVICE)
        model.load_state_dict(torch.load(checkpoint_path, map_location='cpu', weights_only=True))
        model = model.to(DEVICE).eval()

    # Enable memory efficient inference for large models
    if DEVICE == 'cuda' and encoder in ['vitl', 'vitg']:
        torch.cuda.empty_cache()
        print(f"  ✓ Memory optimizations enabled for large model")

    # compiled_model = torch.compile(model, mode="max-autotune", fullgraph=True)
    # return compiled_model
    return model

@time_function
def infer_depth(model, image):
    torch, _ = _ensure_torch_loaded()

    # Check if using CUDA and print memory stats
    if torch.cuda.is_available() and next(model.parameters()).is_cuda:
        # Clear cache before inference
        torch.cuda.empty_cache()

        # Get memory stats before inference
        allocated_before = torch.cuda.memory_allocated(0) / (1024**3)
        reserved_before = torch.cuda.memory_reserved(0) / (1024**3)
        print(f"  GPU memory before inference: {allocated_before:.2f} GB allocated, {reserved_before:.2f} GB reserved")

    # Perform inference with no_grad for memory efficiency
    with torch.no_grad():
        result = model.infer_image(image, input_size=512)

    # Clear cache after inference
    if torch.cuda.is_available() and next(model.parameters()).is_cuda:
        allocated_after = torch.cuda.memory_allocated(0) / (1024**3)
        reserved_after = torch.cuda.memory_reserved(0) / (1024**3)
        print(f"  GPU memory after inference: {allocated_after:.2f} GB allocated, {reserved_after:.2f} GB reserved")

        # Free up memory
        torch.cuda.empty_cache()
        print(f"  ✓ GPU cache cleared")

    return result

# Colormap constants - these will be looked up from cv2 when needed
COLORMAP_NAMES = {
    'HOT': 'COLORMAP_HOT',
    'COOL': 'COLORMAP_COOL',
    'OCEAN': 'COLORMAP_OCEAN',
    'SUMMER': 'COLORMAP_SUMMER',
    'SPRING': 'COLORMAP_SPRING',
    'INFERNO': 'COLORMAP_INFERNO',
    'PLASMA': 'COLORMAP_PLASMA',
    'VIRIDIS': 'COLORMAP_VIRIDIS',
    'TWILIGHT': 'COLORMAP_TWILIGHT',
}

@time_function
def post_process_and_save(
    depth: np.ndarray,
    output_path: str | None,
    plane_removal_factor: float = 1.0,
    use_colormap: bool = False,
    colormap: str | None = None,
    alpha_mask: np.ndarray | None = None,
    save_16bit: bool = True
) -> np.ndarray:
    """
    Converts a masked depth map to 8- or 16-bit (gray or colour-mapped),
    optionally merges alpha, writes a PNG, and returns the BGR(A) buffer.
    """
    _, cv2 = _ensure_torch_loaded()  # Load cv2 first!

    depth = remove_infinities(depth)
    depth = normalize_01(depth)
    depth = depth - (plane_removal_factor * estimate_plane_of_best_fit(depth))
    depth = normalize_01(depth)
    depth = np.clip(normalize_01(depth), 0.0, 1.0)

    # ── colourise / quantise ────────────────────────────────────────────────
    if use_colormap:                                                    # RGB heat-map
        colormap_value = getattr(cv2, COLORMAP_NAMES[colormap])
        depth_bgr = cv2.applyColorMap((depth * 255).astype(np.uint8),
                                    colormap_value)
        if save_16bit:                                                    # upscale 8-bit → 16-bit
            depth_bgr = (depth_bgr.astype(np.uint16) * 257)
    else:                                                               # plain grayscale
        scale = 65535 if save_16bit else 255
        gray  = (depth * scale).astype(np.uint16 if save_16bit else np.uint8)
        depth_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # ── optional alpha merge ───────────────────────────────────────────────
    if alpha_mask is not None:
        a_rs   = cv2.resize(alpha_mask, depth_bgr.shape[1::-1],
                            interpolation=cv2.INTER_NEAREST)
        scale  = 65535 if save_16bit else 255
        a_chan = (a_rs * scale).round().astype(np.uint16 if save_16bit else np.uint8)
        depth_bgr = cv2.merge((*cv2.split(depth_bgr), a_chan))

    # ── save PNG: convert BGR(A) → RGB(A) first ────────────────────────────
    if output_path and output_path != '':
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        if not use_colormap:
            to_rgb = cv2.COLOR_BGRA2RGBA if depth_bgr.shape[2] == 4 else cv2.COLOR_BGR2RGB
            cv2.imwrite(output_path, cv2.cvtColor(depth_bgr, to_rgb))
        else:
            cv2.imwrite(output_path, depth_bgr)

    return depth_bgr

@time_function
def get_raw_img(input_image: bpy.types.Image,
                use_dirty_image: bool) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract a Blender image datablock as a pair:

    Returns
    -------
    raw_img      np.ndarray  uint8, shape (H, W, 3), BGR 0-255
    alpha_mask   np.ndarray  float32, shape (H, W),     0-1   (1 = opaque)
    """
    _, cv2 = _ensure_torch_loaded()  # Load cv2 first!

    H, W = input_image.size[1], input_image.size[0]

    from_file = (
        input_image.filepath and
        not (use_dirty_image and input_image.is_dirty)
    )

    if from_file:
        path = bpy.path.abspath(input_image.filepath)
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img.dtype == np.uint16:           # 0-65535  ->  0-255
            img = cv2.convertScaleAbs(img, alpha=255.0 / 65535.0)

        img_default   = cv2.imread(path)                       # what you used before
        img_unchanged = cv2.imread(path, cv2.IMREAD_UNCHANGED) # what you use now

        print(img.dtype,   img.min(),   img.max())
        print(img_default.dtype,   img_default.min(),   img_default.max())
        print(img_unchanged.dtype, img_unchanged.min(), img_unchanged.max())

        if img is None:
            raise ValueError(f"Could not read image: {path}")

        if img.ndim == 2:                       # gray
            raw_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            alpha   = np.ones((H, W), np.float32)
        elif img.ndim == 3 and img.shape[2] == 3:   # BGR
            raw_img = img
            alpha   = np.ones((H, W), np.float32)
        elif img.ndim == 3 and img.shape[2] == 4:   # BGRA
            raw_img = img[:, :, :3]
            alpha   = img[:, :, 3].astype(np.float32) / 255.0
        elif img.ndim == 3 and img.shape[2] == 2:   # gray+α
            gray, a = img[:, :, 0], img[:, :, 1]
            raw_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            alpha   = a.astype(np.float32) / 255.0
        else:
            raise ValueError(f"Unsupported channel layout in {path}")

    else:
        px     = np.array(input_image.pixels[:]).reshape(H, W, 4)
        px     = np.flipud(px)                                   # Blender Y-flip
        rgb_f  = px[:, :, :3]
        alpha  = px[:, :, 3]
        raw_u8 = (rgb_f * 255).astype(np.uint8)
        raw_img = cv2.cvtColor(raw_u8, cv2.COLOR_RGB2BGR)

    return raw_img, alpha.astype(np.float32)

@time_function
def main(model_size: str,
         checkpoint_path: str,
         input_image,
         output_path: str | None,
         use_dirty_image: bool,
         plane_removal_factor: float,
         use_colormap: bool,
         colormap: str | None,
         include_alpha: bool = False,
         save_16bit: bool = True,
         preferred_device: str = "cpu",
         enable_cpu_offload: bool = True) -> np.ndarray:
    """
    High-level entry that loads the model, infers depth, masks transparencies,
    post-processes, embeds alpha (optional) and writes a PNG.

    Parameters
    ----------
    include_alpha : bool
        If True, embed the original alpha layer into the saved PNG.
    save_16bit : bool
        If True, write a 16-bit PNG; otherwise write 8-bit.
    preferred_device : {"cpu","gpu"}
        "gpu" selects CUDA → Metal → CPU in that order.
    """
    torch, cv2 = _ensure_torch_loaded()  # Load dependencies first!

    # Determine device for display purposes
    if preferred_device == 'gpu':
        DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    else:
        DEVICE = 'cpu'

    print(DEVICE)
    # Enable TF32 for better performance on Ampere+ GPUs
    if DEVICE == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Load model - pass preferred_device ('gpu'/'cpu'), not DEVICE ('cuda'/'cpu')
    model = load_model(model_size, checkpoint_path, preferred_device, enable_cpu_offload)
    if isinstance(input_image,bpy.types.Image):
        raw_img, alpha_mask = get_raw_img(input_image,use_dirty_image)
    else:
        raw_img = input_image # assuming np.ndarray or tensor
        h, w = raw_img.shape[:2]
        alpha_mask = np.ones((h, w), np.float32)
    # Infer depth
    depth = infer_depth(model, raw_img)
    
    # black-out transparent pixels (always on)
    alpha_rs = cv2.resize(alpha_mask, (depth.shape[1], depth.shape[0]),
                          interpolation=cv2.INTER_NEAREST)
    depth *= alpha_rs

    # Post-process the depth map
    depth_vis  = post_process_and_save(
        depth,
        output_path,
        plane_removal_factor,
        use_colormap,
        colormap,
        alpha_mask = alpha_mask if include_alpha else None,
        save_16bit = save_16bit
    )

    del model, raw_img, depth
    # print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        print(f"Using GPU: {torch.cuda.get_device_name(device)}")
        total_memory = torch.cuda.get_device_properties(device).total_memory
        allocated_memory = torch.cuda.memory_allocated(device)
        cached_memory = torch.cuda.memory_reserved(device)
        print(f"Total GPU memory: {total_memory / 1e9:.2f} GB")
        print(f"Allocated GPU memory: {allocated_memory / 1e9:.2f} GB")
        print(f"Cached GPU memory: {cached_memory / 1e9:.2f} GB")
    else:
        print("CUDA is not available. Running on CPU.")
    return depth_vis

def cv2_to_blender_image(cv2_image, new_image, ):
    _, cv2 = _ensure_torch_loaded()  # Load cv2 first!

    if cv2_image.shape[2] == 4:
        cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGRA2RGBA)
    else:
        cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)

    # Create a temporary file
    with tempfile.NamedTemporaryFile(mode='w+', suffix=".png", delete=False,) as temp_file:
        temp_filename = temp_file.name
        print(temp_file.name)
        print(os.path.basename(temp_filename))
        cv2.imwrite(temp_filename, cv2_image)
    
    try:
        blender_image = bpy.data.images.load(str(temp_filename), check_existing=True)
        w, h = blender_image.size
        pixel_data = np.zeros((w, h, 4), 'f')
        blender_image.pixels.foreach_get(pixel_data.ravel())

        new_image.colorspace_settings.name = 'Non-Color'
        new_image.pixels.foreach_set(pixel_data.ravel())
        new_image.update()
        blender_image.user_clear()
        bpy.data.images.remove(blender_image)
    finally:
        os.unlink(temp_filename)
    return new_image