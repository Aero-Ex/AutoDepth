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


def _ensure_torch_loaded():
    """
    Lazy-load torch and cv2 with proper DLL setup for Windows.
    This MUST be called before any function that uses torch/cv2.
    """
    global _torch_loaded, _cv2, _torch

    if _torch_loaded:
        return _torch, _cv2

    print("TrueDepth: Loading PyTorch and OpenCV...")

    # On Windows, we need to add DLL directories BEFORE importing torch
    if sys.platform == "win32" and hasattr(os, 'add_dll_directory'):
        print("TrueDepth: Setting up DLL directories for Windows...")

        # 1. Add Blender's Python DLL directory
        try:
            blender_python_dir = Path(sys.executable).parent
            os.add_dll_directory(str(blender_python_dir))
            print(f"   ✓ Added Blender Python dir: {blender_python_dir}")
        except Exception as e:
            print(f"   ⚠ Could not add Blender Python dir: {e}")

        # 2. Add Blender's parent directory (contains more DLLs)
        try:
            blender_parent = Path(sys.executable).parent.parent
            if blender_parent.exists():
                os.add_dll_directory(str(blender_parent))
                print(f"   ✓ Added Blender parent dir: {blender_parent}")
        except Exception as e:
            print(f"   ⚠ Could not add Blender parent dir: {e}")

        # 3. Add torch's lib directory
        try:
            for path in sys.path:
                torch_lib = Path(path) / "torch" / "lib"
                if torch_lib.exists():
                    os.add_dll_directory(str(torch_lib))
                    print(f"   ✓ Added torch lib dir: {torch_lib}")
                    break
        except Exception as e:
            print(f"   ⚠ Could not add torch lib dir: {e}")

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
        if "DLL" in str(e) and sys.platform == "win32":
            print("\n" + "="*60)
            print("WINDOWS DLL ERROR - POSSIBLE SOLUTIONS:")
            print("1. Install Visual C++ Redistributable:")
            print("   https://aka.ms/vs/17/release/vc_redist.x64.exe")
            print("2. Delete and reinstall dependencies in the addon")
            print("3. Make sure no antivirus is blocking DLL loading")
            print("="*60 + "\n")
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
    def wrapper(model_size, checkpoint_path, preferred_device):
        global _model, _current_device
        
        if _model is None:
            print("Initializing model")
            _model = func(model_size, checkpoint_path, preferred_device)
            _current_device = preferred_device
        elif preferred_device != _current_device:
            print("Rreloading model")
            _model = func(model_size, checkpoint_path, preferred_device)
            _current_device = preferred_device
        else:
            print("Using cached model")
        
        return _model
    
    return wrapper


@time_function
@model_cache_decorator
def load_model(encoder, checkpoint_path, preferred_device):
    torch, _ = _ensure_torch_loaded()  # Load torch first!

    if preferred_device == 'gpu':
        DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    else:
        DEVICE = 'cpu'
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    from .depth_anything_v2 import dpt
    model = dpt.DepthAnythingV2(**model_configs[encoder], device = DEVICE)
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu', weights_only=True))
    model = model.to(DEVICE).eval()
    # compiled_model = torch.compile(model, mode="max-autotune", fullgraph=True)
    # return compiled_model
    return model

@time_function
def infer_depth(model, image):
    return model.infer_image(image, input_size= 512)

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
         preferred_device: str = "cpu") -> np.ndarray:
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

    # Load model
    if preferred_device == 'gpu':
        DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    else:
        DEVICE = 'cpu'

    print(DEVICE)
    # print("CUDA: ", torch.backends.cuda.matmul.allow_tf32)
    # print("CUDNN: ",torch.backends.cudnn.allow_tf32)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    model = load_model(model_size, checkpoint_path,DEVICE)
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