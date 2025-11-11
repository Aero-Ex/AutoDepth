"""
Diagnostic script to check if Visual C++ Redistributable is the issue.
Run this from Blender's Python console or as a standalone script.
"""

import sys
import os
from pathlib import Path
import ctypes
import ctypes.util

print("="*70)
print("VISUAL C++ REDISTRIBUTABLE DIAGNOSTIC")
print("="*70)

# 1. Check if we can load the VC++ runtime DLLs that c10.dll depends on
print("\n1. Testing VC++ Runtime DLLs that PyTorch needs:")
print("-" * 70)

vcruntime_dlls = [
    'vcruntime140.dll',      # Main VC++ 2015-2022 runtime
    'vcruntime140_1.dll',    # Additional VC++ 2017+ runtime
    'msvcp140.dll',          # C++ standard library
    'msvcp140_1.dll',        # C++ standard library (additional)
    'msvcp140_2.dll',        # C++ standard library (additional)
]

all_vcruntime_ok = True
for dll_name in vcruntime_dlls:
    try:
        # Try to load the DLL
        dll_handle = ctypes.CDLL(dll_name)
        print(f"   ✓ {dll_name:25} - FOUND and loadable")
    except OSError as e:
        print(f"   ✗ {dll_name:25} - MISSING or failed: {e}")
        all_vcruntime_ok = False

# 2. Check Windows System32 directory
print("\n2. Checking Windows System32 for VC++ DLLs:")
print("-" * 70)
system32 = Path(os.environ.get('SystemRoot', 'C:\\Windows')) / 'System32'
print(f"System32 path: {system32}")

for dll_name in vcruntime_dlls:
    dll_path = system32 / dll_name
    exists = dll_path.exists()
    status = "✓ EXISTS" if exists else "✗ MISSING"
    size = f"({dll_path.stat().st_size:,} bytes)" if exists else ""
    print(f"   {status} {dll_name:25} {size}")

# 3. Check registry for installed VC++ versions
print("\n3. Checking Windows Registry for VC++ Redistributable:")
print("-" * 70)
try:
    import winreg

    # Check both 64-bit and 32-bit registry paths
    registry_paths = [
        (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\X64"),
        (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\Microsoft\VisualStudio\14.0\VC\Runtimes\X64"),
        (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\X86"),
    ]

    found_any = False
    for hkey, subkey in registry_paths:
        try:
            key = winreg.OpenKey(hkey, subkey)
            try:
                version = winreg.QueryValueEx(key, "Version")[0]
                installed = winreg.QueryValueEx(key, "Installed")[0]
                print(f"   ✓ Found: {subkey}")
                print(f"      Version: {version}, Installed: {installed}")
                found_any = True
            except:
                pass
            winreg.CloseKey(key)
        except FileNotFoundError:
            pass

    if not found_any:
        print("   ✗ No VC++ Redistributable found in registry")

except Exception as e:
    print(f"   ⚠ Could not check registry: {e}")

# 4. Try to load c10.dll directly
print("\n4. Testing c10.dll loading directly:")
print("-" * 70)

# Find torch installation
torch_lib_dir = None
for path in sys.path:
    potential_torch_lib = Path(path) / "torch" / "lib"
    if potential_torch_lib.exists():
        torch_lib_dir = potential_torch_lib
        break

if torch_lib_dir:
    c10_dll_path = torch_lib_dir / "c10.dll"
    print(f"c10.dll path: {c10_dll_path}")
    print(f"c10.dll exists: {c10_dll_path.exists()}")

    if c10_dll_path.exists():
        # Add torch lib to PATH first
        old_path = os.environ.get('PATH', '')
        os.environ['PATH'] = str(torch_lib_dir) + os.pathsep + old_path

        # Also add Blender's python bin directory
        blender_python_dir = Path(sys.executable).parent
        os.environ['PATH'] = str(blender_python_dir) + os.pathsep + os.environ['PATH']

        try:
            # Try to load c10.dll
            print(f"\nAttempting to load c10.dll...")
            c10_handle = ctypes.CDLL(str(c10_dll_path))
            print(f"   ✓ SUCCESS! c10.dll loaded successfully")
            print(f"   → This means all dependencies are available!")
        except OSError as e:
            print(f"   ✗ FAILED to load c10.dll")
            print(f"   Error: {e}")
            print(f"\n   → This error will tell us what's actually missing:")

            # Try to parse the error to see what's missing
            error_str = str(e)
            if "126" in error_str:
                print(f"   → Error 126: The specified module could not be found")
                print(f"   → This usually means a DLL dependency is missing")
            elif "1114" in error_str:
                print(f"   → Error 1114: DLL initialization routine failed")
                print(f"   → This usually means VC++ Redistributable is missing or wrong version")
else:
    print("   ✗ Could not find torch lib directory in sys.path")

# 5. Summary
print("\n" + "="*70)
print("SUMMARY:")
print("="*70)

if all_vcruntime_ok:
    print("✓ All VC++ Runtime DLLs are available and loadable")
    print("  → Visual C++ Redistributable is NOT the problem")
    print("  → The issue is likely something else (PATH, permissions, etc.)")
else:
    print("✗ Some VC++ Runtime DLLs are missing or cannot be loaded")
    print("  → Visual C++ Redistributable IS the problem")
    print("  → Download from: https://aka.ms/vs/17/release/vc_redist.x64.exe")

print("="*70)
