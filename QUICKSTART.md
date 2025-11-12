# TrueDepth - Quick Start Guide

Get up and running with TrueDepth in 5 minutes!

## 🚀 Quick Installation

### 1. Install the Addon (2 minutes)

```
1. Download TrueDepth.zip
2. Open Blender → Edit → Preferences → Add-ons
3. Click "Install..." → Select the ZIP file
4. Enable "TrueDepth" checkbox
```

### 2. Install Dependencies (5-10 minutes)

```
1. In addon preferences, choose device:
   - CPU: Works everywhere (slower)
   - GPU: NVIDIA only (10x faster)

2. Click "Install Dependencies"
3. Wait for installation to complete
4. See "Installation Complete!" message
```

### 3. Generate Your First Depth Map (30 seconds)

```
1. Load any image in Blender
2. Press N → TrueDepth tab
3. Click "Generate Depth Map"
4. Done! Depth map saved to output folder
```

## 💡 Example Workflow

### Using Depth Maps for Displacement

1. Generate depth map from your image
2. Add a plane in Blender
3. Add Subdivision Surface modifier (6+ levels)
4. Add Displace modifier:
   - Texture: Load your depth map
   - Strength: 1.0 (adjust as needed)
5. Result: 3D mesh from your 2D image!

### Using Depth Maps in Compositor

1. Generate depth map
2. Go to Compositing workspace
3. Add Image node with your depth map
4. Use for:
   - Depth of Field effects
   - Fog/atmosphere
   - Z-depth masking
   - Defocus blur

## ⚡ Performance Tips

**For best performance:**
- Use GPU if you have NVIDIA graphics card (10x faster)
- Start with "Small" model (faster, still great quality)
- Process at 1080p or lower for speed
- Use CPU for single images, GPU for batches

**Typical speeds:**
- **CPU**: 10 seconds per image
- **GPU**: 1 second per image

## 🔧 Common Issues & Fixes

### Issue: "DLL initialization failed" (Windows)

**Fix:**
1. Install Visual C++ Redistributable:
   https://aka.ms/vs/17/release/vc_redist.x64.exe
2. Restart Blender
3. Try again

### Issue: GPU not working

**Fix:**
- GPU mode only works with NVIDIA GPUs
- AMD/Intel → Use CPU mode instead
- Update NVIDIA drivers: nvidia.com/drivers

### Issue: Out of memory

**Fix:**
- Use "Small" model instead of "Large"
- Reduce image resolution
- Close other applications

## 📊 What to Expect

| Your Setup | Device | Expected Speed |
|------------|--------|----------------|
| No NVIDIA GPU | CPU | ~10s per image |
| NVIDIA GTX 1060+ | GPU | ~1-2s per image |
| NVIDIA RTX 3060+ | GPU | ~0.5-1s per image |

## 🎯 Next Steps

1. **Read full README.md** for advanced features
2. **Try batch processing** for image sequences
3. **Experiment with models** (small/base/large)
4. **Integrate with your workflow** (displacement, compositing, etc.)

## ❓ Need Help?

- **Troubleshooting**: See README.md "Troubleshooting" section
- **Console errors**: Window → Toggle System Console (shows detailed errors)
- **Report bugs**: Include Blender version, OS, and error messages

---

**That's it! You're ready to create amazing depth maps in Blender!** 🎉
