"""
Test script to identify correct DLL loading strategy for Windows + Blender + PyTorch

This script should be run from Blender's Python console to test DLL loading
"""
import sys
import os
from pathlib import Path

print("=" * 80)
print("BLENDER + PYTORCH DLL INVESTIGATION")
print("=" * 80)

# 1. Blender's Python executable
print(f"\n1. Blender Python executable:")
print(f"   sys.executable = {sys.executable}")

# 2. Blender's DLL directory (should contain python3.dll, etc.)
blender_python_dir = Path(sys.executable).parent
blender_dlls_dir = blender_python_dir.parent / "blender.crt"  # Common location
print(f"\n2. Blender Python directory:")
print(f"   {blender_python_dir}")
print(f"   Contents: {list(blender_python_dir.glob('*.dll'))[:5]}")

if blender_dlls_dir.exists():
    print(f"\n3. Blender CRT DLL directory:")
    print(f"   {blender_dlls_dir}")
    print(f"   Contents: {list(blender_dlls_dir.glob('*.dll'))[:5]}")

# 4. Check sys.path for torch location
print(f"\n4. Looking for torch in sys.path:")
torch_paths = [p for p in sys.path if 'torch' in p.lower() or 'truedepth' in p.lower()]
for p in torch_paths:
    print(f"   {p}")

# 5. Find torch's DLL directory
print(f"\n5. Checking for torch installation:")
for path in sys.path:
    torch_dir = Path(path) / "torch"
    if torch_dir.exists():
        torch_lib = torch_dir / "lib"
        print(f"   Found torch at: {torch_dir}")
        if torch_lib.exists():
            print(f"   Torch lib directory: {torch_lib}")
            dll_files = list(torch_lib.glob('*.dll'))
            print(f"   DLL files ({len(dll_files)}): {[d.name for d in dll_files[:10]]}")

            # Check for c10.dll specifically
            c10_dll = torch_lib / "c10.dll"
            if c10_dll.exists():
                print(f"   ✓ Found c10.dll: {c10_dll}")
                print(f"   Size: {c10_dll.stat().st_size / 1024 / 1024:.2f} MB")
            else:
                print(f"   ✗ c10.dll NOT FOUND!")

# 6. Check if os.add_dll_directory is available
print(f"\n6. DLL loading functions available:")
print(f"   hasattr(os, 'add_dll_directory'): {hasattr(os, 'add_dll_directory')}")

# 7. Try to add DLL directories
if hasattr(os, 'add_dll_directory'):
    print(f"\n7. Testing os.add_dll_directory():")

    # Try Blender's directory
    try:
        os.add_dll_directory(str(blender_python_dir))
        print(f"   ✓ Added: {blender_python_dir}")
    except Exception as e:
        print(f"   ✗ Failed to add {blender_python_dir}: {e}")

    # Try torch lib directory
    for path in sys.path:
        torch_lib = Path(path) / "torch" / "lib"
        if torch_lib.exists():
            try:
                os.add_dll_directory(str(torch_lib))
                print(f"   ✓ Added: {torch_lib}")
            except Exception as e:
                print(f"   ✗ Failed to add {torch_lib}: {e}")
            break

# 8. Try to import torch
print(f"\n8. Attempting to import torch:")
try:
    import torch
    print(f"   ✓ SUCCESS! Torch version: {torch.__version__}")
    print(f"   Torch location: {torch.__file__}")
except Exception as e:
    print(f"   ✗ FAILED: {e}")
    import traceback
    print(traceback.format_exc())

print("\n" + "=" * 80)
print("Investigation complete. See output above.")
print("=" * 80)
