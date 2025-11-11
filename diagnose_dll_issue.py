"""
Advanced DLL diagnostic script for Windows PyTorch loading issues
Run this in Blender's Python console
"""
import sys
import os
from pathlib import Path
import subprocess

print("=" * 80)
print("ADVANCED DLL DIAGNOSTIC FOR PYTORCH")
print("=" * 80)

# 1. Check Python version
print("\n1. PYTHON VERSION CHECK:")
print(f"   Blender Python: {sys.version}")
print(f"   Version info: {sys.version_info}")

# 2. Check if VC++ Redistributable is really installed
print("\n2. VISUAL C++ REDISTRIBUTABLE CHECK:")
if sys.platform == "win32":
    try:
        # Check registry for installed VC++ versions
        import winreg

        reg_paths = [
            r"SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\X64",
            r"SOFTWARE\WOW6432Node\Microsoft\VisualStudio\14.0\VC\Runtimes\X64",
        ]

        found_vcredist = False
        for reg_path in reg_paths:
            try:
                key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, reg_path)
                version = winreg.QueryValueEx(key, "Version")[0]
                print(f"   ✓ Found VC++ Redistributable: {version}")
                found_vcredist = True
                winreg.CloseKey(key)
                break
            except:
                pass

        if not found_vcredist:
            print("   ✗ VC++ Redistributable NOT found in registry!")
            print("   → Download: https://aka.ms/vs/17/release/vc_redist.x64.exe")
    except Exception as e:
        print(f"   ⚠ Could not check registry: {e}")

# 3. Find PyTorch installation
print("\n3. PYTORCH INSTALLATION:")
torch_site_packages = None
for path in sys.path:
    torch_dir = Path(path) / "torch"
    if torch_dir.exists():
        torch_site_packages = Path(path)
        print(f"   ✓ Found torch at: {torch_dir}")

        # Check torch version file
        version_file = torch_dir / "version.py"
        if version_file.exists():
            try:
                with open(version_file) as f:
                    for line in f:
                        if "__version__" in line:
                            print(f"   → {line.strip()}")
            except:
                pass

        # Check for key DLLs
        torch_lib = torch_dir / "lib"
        if torch_lib.exists():
            critical_dlls = ["c10.dll", "torch.dll", "torch_cpu.dll", "libiomp5md.dll"]
            print(f"\n   Critical DLLs in {torch_lib}:")
            for dll_name in critical_dlls:
                dll_path = torch_lib / dll_name
                if dll_path.exists():
                    size_mb = dll_path.stat().st_size / 1024 / 1024
                    print(f"   ✓ {dll_name}: {size_mb:.2f} MB")
                else:
                    print(f"   ✗ {dll_name}: MISSING!")
        break

if not torch_site_packages:
    print("   ✗ PyTorch not found in sys.path!")
    print("\n" + "="*80)
    print("SOLUTION: Install dependencies using the addon's 'Install Dependencies' button")
    print("="*80)
    sys.exit(0)

# 4. Check what c10.dll actually depends on (using Windows dumpbin or Dependencies.exe)
print("\n4. CHECKING DLL DEPENDENCIES:")
print("   Attempting to find what c10.dll depends on...")

c10_dll = torch_site_packages / "torch" / "lib" / "c10.dll"
if c10_dll.exists():
    print(f"   c10.dll location: {c10_dll}")

    # Try using dumpbin if available (comes with Visual Studio)
    try:
        result = subprocess.run(
            ["dumpbin", "/DEPENDENTS", str(c10_dll)],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            print("\n   DLL Dependencies (from dumpbin):")
            for line in result.stdout.split('\n'):
                if '.dll' in line.lower():
                    print(f"      {line.strip()}")
        else:
            print("   ⚠ dumpbin not available (requires Visual Studio)")
    except FileNotFoundError:
        print("   ⚠ dumpbin not found (requires Visual Studio)")
    except Exception as e:
        print(f"   ⚠ Could not run dumpbin: {e}")

# 5. Try to manually load c10.dll to see the exact error
print("\n5. ATTEMPTING TO LOAD C10.DLL DIRECTLY:")
try:
    import ctypes
    c10_handle = ctypes.CDLL(str(c10_dll))
    print("   ✓ SUCCESS! c10.dll loaded directly")
    print("   → The issue is likely in torch's Python bindings, not the DLL itself")
except Exception as e:
    print(f"   ✗ FAILED: {e}")
    print("\n   This is the root cause! c10.dll cannot load.")
    print("   Common reasons:")
    print("   1. Missing VCRUNTIME140.dll (even though VC++ Redist is installed)")
    print("   2. Missing Intel MKL libraries (libmmd.dll, libiomp5md.dll)")
    print("   3. Incompatible Python version")
    print("   4. Corrupted installation")

# 6. Check Python ABI compatibility
print("\n6. PYTHON ABI COMPATIBILITY:")
try:
    # Check if torch was built for this Python version
    torch_lib_dir = torch_site_packages / "torch" / "lib"
    torch_python_dll = None
    for dll in torch_lib_dir.glob("torch_python*.dll"):
        torch_python_dll = dll
        print(f"   Found: {dll.name}")

    # Check Python version in DLL name
    if torch_python_dll:
        if "311" in torch_python_dll.name or "cp311" in torch_python_dll.name:
            if sys.version_info[:2] == (3, 11):
                print("   ✓ PyTorch Python version matches Blender (3.11)")
            else:
                print(f"   ✗ MISMATCH! PyTorch built for Python 3.11, Blender has {sys.version_info[:2]}")
                print("   → This is likely the problem!")
except Exception as e:
    print(f"   ⚠ Could not check Python compatibility: {e}")

# 7. Check PATH environment variable
print("\n7. PATH ENVIRONMENT CHECK:")
path_entries = os.environ.get('PATH', '').split(os.pathsep)
print(f"   Total PATH entries: {len(path_entries)}")

# Check if torch/lib is in PATH
torch_lib_in_path = False
for entry in path_entries[:10]:  # Show first 10
    if 'torch' in entry.lower() and 'lib' in entry.lower():
        print(f"   ✓ Torch lib in PATH: {entry}")
        torch_lib_in_path = True
        break

if not torch_lib_in_path:
    print("   ⚠ Torch lib directory NOT in PATH")
    print("   → The addon should add this automatically when loading")

# 8. Final recommendations
print("\n" + "="*80)
print("DIAGNOSTIC COMPLETE - RECOMMENDATIONS:")
print("="*80)

print("\nIf c10.dll failed to load directly (test #5), try these IN ORDER:")
print("\n1. REINSTALL PYTORCH:")
print("   a) Delete: C:\\Users\\Admin\\TrueDepth-blender-addon\\python3.11\\venv_depthanything_cpu")
print("   b) In Blender addon preferences, click 'Install Dependencies'")
print("   c) Wait for completion, then restart Blender")

print("\n2. CHECK FOR DLL CONFLICTS:")
print("   a) Download Dependencies.exe: https://github.com/lucasg/Dependencies/releases")
print("   b) Open c10.dll in Dependencies.exe")
print("   c) Look for missing/red DLLs in the dependency tree")
print("   d) Screenshot and share if you need help")

print("\n3. TRY INSTALLING INTEL MKL:")
print("   If libiomp5md.dll is the issue, you may need Intel MKL")
print("   Download: https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-download.html")

print("\n4. USE DEPENDENCIES.EXE TO INSPECT:")
print(f"   Load this file: {c10_dll}")
print("   Look at the dependency tree to see what's missing")

print("\n" + "="*80)
