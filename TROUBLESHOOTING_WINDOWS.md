# Windows Troubleshooting Guide

Detailed troubleshooting guide for Windows-specific issues with TrueDepth addon.

## Table of Contents
1. [DLL Errors](#dll-errors)
2. [Installation Issues](#installation-issues)
3. [GPU Issues](#gpu-issues)
4. [Performance Issues](#performance-issues)
5. [Diagnostic Tools](#diagnostic-tools)

---

## DLL Errors

### Error: "WinError 1114: A dynamic link library (DLL) initialization routine failed"

This is the most common error on Windows. The addon includes automatic fixes, but here's what to do if you still see this:

#### Step 1: Verify Visual C++ Redistributable

Even though the addon works without it in most cases, verify you have it:

1. **Check if installed**:
   - Open "Add or Remove Programs"
   - Search for "Microsoft Visual C++ 2015-2022 Redistributable"
   - Should see both x64 and x86 versions

2. **Install/Repair if needed**:
   - Download: https://aka.ms/vs/17/release/vc_redist.x64.exe
   - Run the installer
   - Choose "Repair" if already installed
   - Restart your computer

#### Step 2: Run Diagnostic Script

1. Open Blender
2. Go to Scripting workspace
3. Open the Python Console (bottom panel)
4. Run this code:

```python
import sys
sys.path.append(r"C:\Users\YOUR_USERNAME\AppData\Roaming\Blender Foundation\Blender\4.5\extensions\user_default\truedepth")
exec(open(r"C:\Users\YOUR_USERNAME\AppData\Roaming\Blender Foundation\Blender\4.5\extensions\user_default\truedepth\check_vcredist.py").read())
```

5. **Check the output**:
   - If all VC++ DLLs show ✓ (green): VC++ is NOT the problem
   - If any show ✗ (red): Install/repair VC++ Redistributable

#### Step 3: Reinstall with Correct PyTorch Version

The addon now uses PyTorch 2.1.0 (stable) instead of 2.9.0 (buggy):

1. **Delete old installation**:
   ```
   C:\Users\YOUR_USERNAME\TrueDepth-blender-addon\python3.11\venv_depthanything_cpu
   ```

2. **Restart Blender completely** (important!)

3. **Reinstall dependencies**:
   - Open addon preferences
   - Click "Install Dependencies"
   - Wait for completion

4. **Verify version**:
   - After installation, generate a depth map
   - Console should show: "Successfully loaded PyTorch 2.1.0+cpu"
   - If it shows 2.9.0, delete and reinstall again

#### Step 4: Check Windows Defender / Antivirus

Sometimes security software blocks DLL loading:

1. **Temporarily disable** Windows Defender / antivirus
2. **Try generating** a depth map
3. If it works:
   - Add exceptions for:
     - Blender.exe
     - Your TrueDepth-blender-addon folder
   - Re-enable security software

### Error: "Error loading c10.dll or one of its dependencies"

This typically means the PyTorch installation is corrupted.

**Fix**:
1. Close Blender
2. Delete the entire venv folder:
   ```
   C:\Users\YOUR_USERNAME\TrueDepth-blender-addon\python3.11
   ```
3. Restart Blender
4. Reinstall dependencies (will download fresh copies)

---

## Installation Issues

### Issue: Installation Hangs or Takes Forever

**Symptoms**:
- Install button clicked but nothing happens
- Console shows "Installing..." for 30+ minutes
- No progress indication

**Fixes**:

1. **Check Internet Connection**:
   - PyTorch is 1.5GB download
   - Slow internet = slow install
   - Check: Can you browse the web normally?

2. **Check Disk Space**:
   - Need 5GB free space
   - Check: Right-click C: drive → Properties

3. **Retry Installation**:
   - Close the console window
   - In Blender, disable and re-enable the addon
   - Try "Install Dependencies" again

4. **Check Firewall**:
   - Windows Firewall may block Python
   - Allow Python.exe through firewall

### Issue: "Failed to create virtual environment"

**Symptoms**:
- Error during installation start
- Can't create venv folder

**Fixes**:

1. **Check Folder Permissions**:
   - Right-click the dependencies folder
   - Properties → Security
   - Make sure your user has "Full Control"

2. **Use Different Location**:
   - In addon preferences, change "Dependencies Path"
   - Choose a folder you definitely have write access to
   - Example: `C:\TrueDepth-Data`

3. **Run Blender as Administrator** (temporary test):
   - Right-click Blender shortcut
   - "Run as Administrator"
   - Try installation again
   - If it works, it's a permissions issue

### Issue: Installation Completes but Addon Doesn't Work

**Symptoms**:
- Installation says "Complete"
- But depth map generation fails
- Import errors in console

**Fixes**:

1. **Restart Blender Completely**:
   - Close all Blender windows
   - Wait 5 seconds
   - Open Blender again
   - Try the addon

2. **Check Installation Marker**:
   - Look for this file:
     ```
     C:\Users\YOUR_USERNAME\TrueDepth-blender-addon\python3.11\installation_complete_cpu.txt
     ```
   - If missing, installation didn't actually complete
   - Reinstall dependencies

3. **Verify File Sizes**:
   - torch folder should be ~1.5GB
   - If much smaller, download was incomplete
   - Delete and reinstall

---

## GPU Issues

### Issue: GPU Not Detected

**Symptoms**:
- Selected GPU mode but it runs on CPU
- Console shows "CUDA is not available"

**Diagnosis**:

1. **Check if you have NVIDIA GPU**:
   - Press `Win + X` → Device Manager
   - Expand "Display adapters"
   - Must say "NVIDIA" - AMD/Intel won't work

2. **Check CUDA compatibility**:
   - GPU must be GTX 900 series or newer
   - Check: https://developer.nvidia.com/cuda-gpus
   - Find your GPU in the list

3. **Update NVIDIA Drivers**:
   - Download latest from: https://www.nvidia.com/download/index.aspx
   - Select your GPU model
   - Install and restart computer

4. **Test CUDA in Blender Console**:
   ```python
   import sys
   sys.path.insert(0, r"C:\Users\YOUR_USERNAME\TrueDepth-blender-addon\python3.11\venv_depthanything_gpu\Lib\site-packages")
   import torch
   print(f"CUDA available: {torch.cuda.is_available()}")
   print(f"CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
   ```

**Fixes**:

1. **If you have AMD/Intel GPU**:
   - GPU acceleration not possible (PyTorch limitation)
   - Use CPU mode instead
   - Still works, just slower

2. **If you have old NVIDIA GPU** (pre-GTX 900):
   - GPU too old for CUDA 11.8
   - Use CPU mode instead

3. **If you have compatible NVIDIA GPU**:
   - Update drivers: https://nvidia.com/drivers
   - Reinstall GPU dependencies
   - Restart computer

### Issue: "CUDA out of memory"

**Symptoms**:
- GPU mode works but crashes with out of memory
- Happens on large images or large models

**Fixes**:

1. **Use Smaller Model**:
   - Switch from "Large" to "Base" or "Small"
   - Small model uses ~2GB VRAM vs 4GB for Large

2. **Reduce Image Size**:
   - Scale down images before processing
   - 1080p instead of 4K
   - Or process tiles instead of full image

3. **Close Other GPU Applications**:
   - Close games, 3D apps, video editors
   - Chrome can use lots of GPU memory (hardware acceleration)
   - Free up VRAM for depth map generation

4. **Switch to CPU Mode**:
   - If you have 4GB or less VRAM
   - CPU mode has no memory limit (uses RAM instead)

---

## Performance Issues

### Issue: Slow Depth Map Generation

**Expected Performance**:
- CPU: 7-15 seconds per image (depending on model size)
- GPU: 1-3 seconds per image (depending on model size)

**If much slower**:

1. **Check CPU/GPU Usage**:
   - Open Task Manager (Ctrl+Shift+Esc)
   - Check Performance tab
   - CPU should be 100% (CPU mode) or GPU 100% (GPU mode)

2. **Close Other Applications**:
   - Free up CPU/RAM/GPU resources
   - Especially: Chrome, video editors, games

3. **Check Thermal Throttling**:
   - Laptop getting hot? May slow down to cool off
   - Use cooling pad
   - Clean dust from vents

4. **Try Smaller Model**:
   - "Small" model is 3x faster than "Large"
   - Quality difference is minimal for many uses

5. **Reduce Image Resolution**:
   - Processing 4K takes 4x longer than 1080p
   - Downscale images first if possible

### Issue: Blender Freezes During Generation

**Symptoms**:
- Blender becomes unresponsive
- Can't click anything
- Appears frozen

**This is Normal!**:
- Depth map generation blocks the UI
- Blender will unfreeze when done
- Watch the console for progress

**If it freezes for 5+ minutes**:
1. Check Task Manager - is Python using CPU/GPU?
2. If yes, just wait (processing large image)
3. If no, something crashed - check console for errors

---

## Diagnostic Tools

### Tool 1: Visual C++ Redistributable Check

**File**: `check_vcredist.py`

**What it does**:
- Checks if VC++ runtime DLLs are installed
- Tests if they can be loaded
- Checks Windows registry
- Attempts to load c10.dll directly

**How to run**:

From Windows PowerShell:
```powershell
cd "C:\Users\YOUR_USERNAME\AppData\Roaming\Blender Foundation\Blender\4.5\extensions\user_default\truedepth"
& "C:\Users\YOUR_USERNAME\Documents\blender-4.5.4-windows-x64\blender-4.5.4-windows-x64\4.5\python\bin\python.exe" check_vcredist.py
```

**What to look for**:
- All VC++ DLLs show ✓: VC++ is fine
- Any ✗ marks: Install VC++ Redistributable
- c10.dll loads successfully: PyTorch installation is good
- c10.dll fails: Reinstall dependencies

### Tool 2: Test Dependencies

**Location**: Addon preferences → "Test Dependencies" button

**What it does**:
- Attempts to import torch and cv2
- Reports version numbers
- Checks CUDA availability (GPU mode)

**How to use**:
1. Open addon preferences
2. Click "Test Dependencies"
3. Check the popup message

**Interpreting results**:
- ✓ Success message with versions: Everything works!
- Import Error: Dependencies not installed or corrupted
- DLL Error: See DLL Errors section above

### Tool 3: Console Output

**How to view**:
- Windows → Toggle System Console (in Blender)
- Console shows detailed diagnostic output

**Important messages**:

```
✓ Pre-loaded python311.dll     → Good!
✓ Pre-loaded: c10.dll           → Good!
✓ Successfully loaded PyTorch   → Good!
✗ FAILED to load dependencies   → Problem!
⚠ Could not pre-load c10.dll    → Problem!
```

**If you see errors**:
- Copy the full error message
- Check the specific error section in this guide
- Report issue with full console output if not resolved

---

## Still Need Help?

If you've tried everything and still have issues:

1. **Collect Information**:
   - Windows version (Win+R → `winver`)
   - Blender version
   - GPU model (if using GPU mode)
   - Full console output with error
   - Screenshot of the error

2. **Check README**:
   - Read the full README.md
   - Check FAQ section

3. **Report Issue**:
   - Include all information from step 1
   - Describe what you tried
   - Mention which troubleshooting steps you followed

---

## Summary of Solutions

| Problem | Quick Fix |
|---------|-----------|
| WinError 1114 | Delete venv folder, restart Blender, reinstall |
| Import errors | Restart Blender completely |
| Installation hangs | Check internet and disk space |
| GPU not detected | Update NVIDIA drivers, verify GPU compatibility |
| Out of memory | Use smaller model or reduce image size |
| Slow performance | Close other apps, use GPU mode if available |

**Most Common Solution**: Delete the venv folder and reinstall dependencies after restarting Blender.

---

**Remember**: The addon now includes automatic DLL loading fixes. Most issues can be resolved by ensuring you have the stable PyTorch 2.1.0 version installed (not 2.9.0).
