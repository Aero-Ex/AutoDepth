# TrueDepth - Blender Addon for AI Depth Map Generation

Generate high-quality depth maps from images using AI (Depth Anything V2 model) directly in Blender.

![Blender Version](https://img.shields.io/badge/Blender-4.5%2B-orange)
![Python Version](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch Version](https://img.shields.io/badge/PyTorch-2.1.0-red)
![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey)

## Features

- 🎨 Generate depth maps from any image directly in Blender
- 🚀 GPU acceleration support (3-10x faster than CPU)
- 🔄 Automatic dependency installation
- 💾 Batch processing support for image sequences
- 🎯 Uses state-of-the-art Depth Anything V2 model
- ⚡ Optimized for Windows with proper DLL handling

## System Requirements

### Minimum Requirements
- **Blender**: 4.5 or newer
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 5GB free space for dependencies
- **OS**: Windows 10/11, macOS 10.15+, or Linux

### For GPU Acceleration (Optional)
- **GPU**: NVIDIA GPU with CUDA support (GeForce GTX 900 series or newer)
- **VRAM**: 4GB minimum, 6GB+ recommended
- **Driver**: NVIDIA driver 452.39 or newer

### Windows-Specific Requirements
- **Visual C++ Redistributable 2015-2022** (usually already installed)
  - Download: https://aka.ms/vs/17/release/vc_redist.x64.exe
  - Note: Most Windows systems already have this installed

## Installation

### Step 1: Download the Addon

1. Download the addon as a ZIP file
2. **Do NOT extract the ZIP file** - Blender will handle this

### Step 2: Install in Blender

1. Open Blender
2. Go to **Edit → Preferences → Add-ons**
3. Click **Install...** button (top right)
4. Navigate to the downloaded ZIP file and select it
5. Click **Install Add-on**
6. Enable the addon by checking the box next to "TrueDepth"

### Step 3: Configure Installation Path

1. In the TrueDepth addon preferences, you'll see:
   - **Dependencies Path**: Where PyTorch and other libraries will be installed
   - Default location is fine for most users (~3-5GB)

### Step 4: Install Dependencies

1. Choose your device:
   - **CPU**: Works on all systems, slower but reliable
   - **GPU**: Requires NVIDIA GPU, 3-10x faster

2. Click **"Install Dependencies"** button

3. A console window will open showing installation progress:
   ```
   Initializing Virtual Environment...
   Installing PyTorch 2.1.0 (stable, tested version)...
   ```

4. Wait for installation to complete (5-15 minutes depending on internet speed)

5. When you see:
   ```
   ╔═══════════════════════════════════════════════╗
   ║     Installation Complete!                    ║
   ╚═══════════════════════════════════════════════╝
   ```
   You're ready to use TrueDepth!

## Usage

### Basic Usage

1. **Open Blender** and create a new scene or open an existing one

3. **Generate Depth Map**:
   - In the **3D Viewport**, press `N` to open the sidebar
   - Navigate to the **TrueDepth** tab
   - Configure settings:
     - **Model**: Choose model size (small/base/large)
     - **choose image**: choose the preffered image
     - **Output Path**: Where depth maps will be saved
   - Click **"Generate Depth Map"**

4. **View Results**:
   - Depth maps are saved to your specified output path
   - Click on the Create Mesh button to generate the displaced mesh

### Batch Processing

For processing multiple images or animation sequences:

1. Set up your image sequence in Blender
2. In TrueDepth settings:
   - Enable **"Batch Mode"**
   - Set **Start Frame** and **End Frame**
3. Click **"Generate Depth Maps"**
4. Depth maps will be generated for each frame

### Performance Tips

**CPU Mode:**
- Processing time: ~9-10 seconds per frame
- Works on all systems
- Good for single images or small batches

**GPU Mode:**
- Processing time: ~1-3 seconds per frame
- Requires NVIDIA GPU
- Ideal for batch processing or real-time workflows

**Model Size:**
- **Small**: Fastest, good quality (~2GB VRAM)
- **Base**: Balanced speed/quality (~3GB VRAM)
- **Large**: Best quality, slower (~4GB VRAM)

## Troubleshooting

### Windows DLL Errors (WinError 1114)

**Symptom**: Error message about "DLL initialization routine failed"

**Solution**: This addon includes automatic fixes, but if you still encounter this:

1. Verify Visual C++ Redistributable is installed:
   - Download and install: https://aka.ms/vs/17/release/vc_redist.x64.exe
   - Restart Blender after installation

2. If issue persists, reinstall dependencies:
   - Delete the dependencies folder (shown in addon preferences)
   - Restart Blender
   - Click "Install Dependencies" again

### GPU Not Detected

**Symptom**: Addon runs on CPU even though you selected GPU

**Check:**
1. **NVIDIA GPU?** AMD/Intel GPUs are not supported by PyTorch CUDA
2. **Updated drivers?** Download latest from nvidia.com
3. **CUDA compatibility?** GPU must support CUDA 11.8 (GTX 900 series or newer)

**Test GPU:**
- In addon preferences, click "Test Dependencies"
- Console should show: "CUDA available: True"

### Import Errors

**Symptom**: "Failed to import dependencies" error

**Solutions:**
1. Close and restart Blender completely
2. Disable and re-enable the addon
3. Check console for specific error messages
4. Reinstall dependencies (delete folder → reinstall)

### Slow Performance

**Symptom**: Depth map generation takes very long

**Solutions:**
1. **Use GPU mode** if you have NVIDIA GPU (10x faster)
2. **Choose smaller model** (small vs base vs large)
3. **Reduce image resolution** before processing
4. **Close other applications** to free up RAM

### Out of Memory Errors

**Symptom**: "CUDA out of memory" or system slowdown

**Solutions:**
1. **Use smaller model** (small uses ~2GB VRAM vs large using ~4GB)
2. **Reduce image resolution**
3. **Close other GPU-intensive applications**
4. **Switch to CPU mode** if GPU VRAM is insufficient

## Technical Details

### Architecture

- **AI Model**: Depth Anything V2 (state-of-the-art monocular depth estimation)
- **Framework**: PyTorch 2.1.0 (stable, tested version)
- **Backend**: OpenCV for image processing
- **Inference**: Optimized for both CPU and GPU

### Why PyTorch 2.1.0?

This addon uses PyTorch 2.1.0 instead of the latest version because:
- ✓ Proven stability on Windows systems
- ✓ Proper DLL initialization (newer versions have Windows bugs)
- ✓ Full compatibility with Blender's Python 3.11
- ✓ Extensive testing with Depth Anything V2 models

### DLL Loading on Windows

The addon includes sophisticated DLL pre-loading to solve common Windows issues:
- Pre-loads `python311.dll` to ensure PyTorch extensions can find it
- Pre-loads torch core DLLs (`c10.dll`, `torch_cpu.dll`, etc.) in correct order
- Pre-loads CUDA DLLs for GPU builds
- Configures DLL search paths automatically

### File Structure

```
TrueDepth-blender-addon/
├── __init__.py              # Main addon registration
├── operators.py             # Blender operators
├── preferences.py           # Addon preferences UI
├── install_packages.py      # Dependency installer
├── depth_estimation.py      # Core depth estimation logic
├── utils.py                 # Helper utilities
├── plane_fit.py             # Plane fitting algorithms
└── python3.11/              # Dependencies installed here
    └── venv_depthanything_cpu/   # CPU version
    └── venv_depthanything_gpu/   # GPU version (if installed)
```

## Performance Benchmarks

Tested on a typical system (Intel i7-10700, 16GB RAM):

| Mode | Model Size | Image Size | Time per Frame | VRAM/RAM Usage |
|------|-----------|-----------|----------------|----------------|
| CPU  | Small     | 1920x1080 | ~7s            | ~4GB RAM       |
| CPU  | Base      | 1920x1080 | ~10s           | ~5GB RAM       |
| CPU  | Large     | 1920x1080 | ~15s           | ~6GB RAM       |
| GPU  | Small     | 1920x1080 | ~0.8s          | ~2GB VRAM      |
| GPU  | Base      | 1920x1080 | ~1.2s          | ~3GB VRAM      |
| GPU  | Large     | 1920x1080 | ~2.5s          | ~4GB VRAM      |

*Your results may vary based on hardware*

## FAQ

**Q: Can I use AMD or Intel GPUs?**
A: No, GPU acceleration requires NVIDIA GPUs with CUDA support. AMD/Intel GPUs will automatically fall back to CPU mode.

**Q: Does this work with Blender 3.x?**
A: The addon is designed for Blender 4.5+. It may work on 4.0+, but is untested on Blender 3.x.

**Q: Can I use this for commercial projects?**
A: Yes, the addon and depth maps you generate can be used commercially. Check the Depth Anything V2 model license for AI model usage terms.

**Q: Why does installation take so long?**
A: PyTorch and dependencies are large (3-5GB). The installer downloads and sets up everything needed, which takes 5-15 minutes depending on internet speed.

**Q: Can I install dependencies manually?**
A: Not recommended. The addon uses a specific virtual environment structure and PyTorch version. Use the built-in installer for best results.

**Q: Does this send data to the cloud?**
A: No, everything runs locally on your machine. No internet connection is needed after initial installation.

**Q: What image formats are supported?**
A: All formats supported by Blender/OpenCV: PNG, JPG, JPEG, TGA, BMP, TIFF, EXR, etc.

## Credits

- **Depth Anything V2**: LiheYoung, Bingyi Kang, Zilong Huang, et al.
- **PyTorch**: Facebook AI Research
- **OpenCV**: Open Source Computer Vision Library
- **Addon Development**: Built for the Blender community

## License

This addon is released under the MIT License. See LICENSE file for details.

The Depth Anything V2 model has its own license - please check the official repository for AI model usage terms.

## Support

If you encounter issues:

1. **Check Troubleshooting section** above
2. **Check console output** (Window → Toggle System Console)
3. **Report issues** with:
   - Blender version
   - Operating system
   - Error messages from console
   - Steps to reproduce

## Changelog

### Version 1.0.0 (2025-01)
- Initial release
- Windows DLL loading fixes
- PyTorch 2.1.0 stable version
- CPU and GPU support
- Depth Anything V2 integration
- Batch processing support

---

**Enjoy creating depth maps in Blender!** 🎨
