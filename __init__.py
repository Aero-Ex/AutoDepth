# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTIBILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

bl_info = {
    "name": "TrueDepth",
    "author": "Pavan, Lewis",
    "description": "Generate depthmap in seconds using AI",
    "blender": (2, 80, 0),
    "version": (1, 5, 4),
    "location": "View3D > Sidebar > TrueDepth",
    "warning": "",
    "category": "3D View",
}

import sys
import json
from pathlib import Path

# Blender imports
import bpy

# Global settings
DEFAULT_SETTINGS = {
    "device": "cpu",
    "dependency_path": None
}
SETTINGS_FILE = "settings.json"

# Create data directory
dg_path = Path("~/TrueDepth-blender-addon/").expanduser().resolve()
dg_path.mkdir(exist_ok=True)

pythonversion = f"python{sys.version_info.major}.{sys.version_info.minor}"
(dg_path / pythonversion).mkdir(exist_ok=True)

settings_filepath = dg_path / SETTINGS_FILE


def load_settings(settings_filepath: Path):
    """Load settings from JSON file"""
    if settings_filepath.exists() and settings_filepath.is_file():
        try:
            with open(settings_filepath, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"TrueDepth: Failed to load settings: {e}")
            return DEFAULT_SETTINGS.copy()
    return DEFAULT_SETTINGS.copy()


def save_settings(settings_filepath: Path, settings: dict):
    """Save settings to JSON file"""
    try:
        with open(settings_filepath, 'w') as f:
            json.dump(settings, f, indent=2)
    except Exception as e:
        print(f"TrueDepth: Failed to save settings: {e}")


# Load settings
DEFAULT_SETTINGS.update(load_settings(settings_filepath))
print(f"TrueDepth: Using device: {DEFAULT_SETTINGS['device']}")

# Import install_packages to ensure dependencies are in path
from . import install_packages

try:
    install_packages.ensure_package_path(
        DEFAULT_SETTINGS['device'],
        target=DEFAULT_SETTINGS["dependency_path"]
    )
except Exception as e:
    import traceback
    print(f"TrueDepth: Error ensuring package path:")
    print(traceback.format_exc())

# Import addon modules
from . import preferences
from . import operators
from . import image_plane
from . import ui
from . import generate_depthmap_video
from . import generate_depthmap_batch


def validate_save_location(self, context):
    """Validate the save location and show warnings if needed"""
    depthgenius = context.scene.depthgenius
    depthgenius.warning_message = ""

    if depthgenius.save_location == 'BLEND_FILE' and not bpy.data.is_saved:
        depthgenius.warning_message = "Warning: Blend file is not saved. Please save the file or choose a different save location."
    elif depthgenius.save_location == 'ORIGINAL_IMAGE' and (not depthgenius.image or not depthgenius.image.filepath):
        depthgenius.warning_message = "Warning: Selected image has no filepath. Please save the image or choose a different save location."
    elif depthgenius.save_location == 'CUSTOM_DIR' and not depthgenius.custom_save_dir:
        depthgenius.warning_message = "Warning: Custom save directory is not specified."
    elif depthgenius.save_location == 'NO_SAVE' and (depthgenius.image and depthgenius.mode == 'MOVIE'):
        depthgenius.warning_message = "Warning: This method is not supported for videos."


def check_model_availability(self, context):
    """Check if the selected model is available"""
    exists, path = operators.checkpoint_exits(self.model_size)
    self.current_model_avaibility = exists


class DEPTHGENIUS_PG_progress(bpy.types.PropertyGroup):
    """Property group for tracking progress"""
    progress: bpy.props.FloatProperty(
        default=0.0,
        min=0.0,
        max=100.0,
        subtype='PERCENTAGE'
    )
    status: bpy.props.StringProperty(default="")
    is_running: bpy.props.BoolProperty(default=False)


class DepthGeniusProperties(bpy.types.PropertyGroup):
    """Main property group for TrueDepth addon"""

    model_size: bpy.props.EnumProperty(
        name="Model Size",
        description="Size of the Depth Anything model",
        items=[
            ('vits', "ViT-S : Small", "Small model (commercial use allowed)"),
            ('vitb', "ViT-B : Base", "Base model (non-commercial use only)"),
            ('vitl', "ViT-L : Large", "Large model (non-commercial use only)"),
        ],
        default='vits',
        update=check_model_availability
    )

    image: bpy.props.PointerProperty(
        name="Image",
        type=bpy.types.Image,
        description="Image to generate depth map from",
        update=validate_save_location
    )

    depth_map: bpy.props.PointerProperty(
        name="Depth Map",
        type=bpy.types.Image,
        description="Generated depth map"
    )

    displacement_method: bpy.props.EnumProperty(
        name="Displacement Method",
        description="Method to apply displacement",
        items=[
            ('MESH', "Mesh Displacement", "Apply displacement as a mesh modifier", 'MOD_DISPLACE', 0),
            ('MATERIAL', "Material Displacement", "Apply displacement as a material with adaptive subdivision", 'MATERIAL', 1),
        ],
        default='MESH'
    )

    displacement_strength: bpy.props.FloatProperty(
        name="Displacement Strength",
        description="Strength of the displacement effect",
        default=0.1,
        min=0.0,
        max=1.0
    )

    save_location: bpy.props.EnumProperty(
        name="Save Location",
        description="Where to save the generated depth map",
        items=[
            ('BLEND_FILE', "Beside Blend File", "Save beside the current Blender file", 'FILE_BLEND', 0),
            ('ORIGINAL_IMAGE', "Beside Original Image", "Save beside the original image", 'FILE_IMAGE', 1),
            ('CUSTOM_DIR', "Custom Directory", "Save in a specified directory", 'FILE_FOLDER', 2),
            ('NO_SAVE', "Don't Save", "Don't save to disk, only store in Blender", 'CON_TRANSFORM_CACHE', 3),
        ],
        default='ORIGINAL_IMAGE',
        update=validate_save_location
    )

    custom_save_dir: bpy.props.StringProperty(
        name="Custom Save Directory",
        description="Custom directory to save the depth map",
        default="",
        subtype='DIR_PATH',
        update=validate_save_location
    )

    warning_message: bpy.props.StringProperty(
        name="Warning Message",
        description="Warning message for edge cases",
        default=""
    )

    use_dirty_image: bpy.props.BoolProperty(
        name="Use Edited Image Data",
        description="Use the current pixel data of the image if it has been edited",
        default=True
    )

    installation_in_progress: bpy.props.BoolProperty(
        name="Installation In Progress?",
        description="This flag is set when installation script is running",
        default=False
    )

    current_model_avaibility: bpy.props.BoolProperty(
        name="Model Available?",
        description="Current selected model is available or not",
        default=True
    )

    plane_removal_factor: bpy.props.FloatProperty(
        name="Tilt-Correction/Un-Distort",
        description="Play with value to understand the effect. Try with image with wall or flat surface",
        default=0.0,
        soft_min=-1.0,
        soft_max=2.0,
    )

    frame_step: bpy.props.IntProperty(
        name="Frame Step",
        description="Process every nth frame",
        default=1,
        min=1
    )

    save_as_video: bpy.props.BoolProperty(
        name="Save as Video",
        description="Save output as video file (MP4) instead of separate image frames",
        default=True
    )

    device: bpy.props.EnumProperty(
        name="Inference Device",
        description="Device used to compute",
        items=[
            ('cpu', "CPU", "Use CPU for depth estimation"),
            ('gpu', "GPU", "Use GPU for depth estimation"),
        ],
        default=DEFAULT_SETTINGS['device']
    )

    mode: bpy.props.EnumProperty(
        items=[
            ('IMAGE', 'Single Image', 'Depth Image generated from a single image'),
            ('MOVIE', 'Video File', 'Generate Depth Video from selected video file'),
            ('BATCH_IMG', 'Batch Images', 'Batch generate Depth Images from multiple images from single run')
        ],
        description='Input mode',
        name='Mode',
        default='IMAGE',
        update=validate_save_location
    )

    progress: bpy.props.PointerProperty(type=DEPTHGENIUS_PG_progress)

    use_colormap: bpy.props.BoolProperty(
        default=False,
        name="Use Colormap"
    )

    colormaps = [
        ('HOT', 'HOT', ''),
        ('COOL', 'COOL', ''),
        ('OCEAN', 'OCEAN', ''),
        ('SUMMER', 'SUMMER', ''),
        ('SPRING', 'SPRING', ''),
        ('INFERNO', 'INFERNO', ''),
        ('PLASMA', 'PLASMA', ''),
        ('VIRIDIS', 'VIRIDIS', ''),
        ('TWILIGHT', 'TWILIGHT', '')
    ]
    colormap: bpy.props.EnumProperty(items=colormaps, name="Colormap")

    include_alpha: bpy.props.BoolProperty(
        name="Include Alpha Channel",
        description="Copies the source image's alpha into the exported depth map",
        default=False
    )


# Classes to register
classes = (
    preferences.DEPTHGENIUS_Preferences,
    install_packages.DEPTHGENIUS_OT_InstallDependencies,
    DEPTHGENIUS_PG_progress,
    DepthGeniusProperties,
    operators.DEPTHGENIUS_OT_GenerateDepthMap,
    generate_depthmap_video.DEPTHGENIUS_OT_GenerateVideoDepthMap,
    operators.DEPTHGENIUS_OT_open_image_in_new_window,
    image_plane.DEPTHGENIUS_OT_CreatePlane,
    operators.DEPTHGENIUS_OT_open_checkpoint_folder,
    ui.DEPTHGENIUS_PT_HoverInfo,
    ui.DEPTHGENIUS_PT_Panel,
    ui.DEPTHGENIUS_PT_Modifiers,
    ui.DEPTHGENIUS_PT_Info,
)


def register():
    """Register addon classes and properties"""
    print("TrueDepth: Registering addon...")

    # Register all classes
    for cls in classes:
        try:
            bpy.utils.register_class(cls)
        except Exception as e:
            print(f"TrueDepth: Failed to register {cls.__name__}: {e}")

    # Register batch generation
    try:
        generate_depthmap_batch.register()
    except Exception as e:
        print(f"TrueDepth: Failed to register batch generation: {e}")

    # Add property to Scene
    bpy.types.Scene.depthgenius = bpy.props.PointerProperty(type=DepthGeniusProperties)

    print("TrueDepth: Registration complete")


def unregister():
    """Unregister addon classes and properties"""
    print("TrueDepth: Unregistering addon...")

    # Save settings
    try:
        settings = DEFAULT_SETTINGS.copy()
        if hasattr(bpy.context.scene, 'depthgenius'):
            settings["device"] = bpy.context.scene.depthgenius.device
            if hasattr(bpy.context.preferences.addons.get(__package__), 'preferences'):
                settings["dependency_path"] = str(
                    Path(bpy.context.preferences.addons[__package__].preferences.dependencies_path)
                )
        save_settings(settings_filepath, settings)
    except Exception as e:
        print(f"TrueDepth: Failed to save settings: {e}")

    # Unregister batch generation
    try:
        generate_depthmap_batch.unregister()
    except Exception as e:
        print(f"TrueDepth: Failed to unregister batch generation: {e}")

    # Unregister all classes
    for cls in reversed(classes):
        try:
            bpy.utils.unregister_class(cls)
        except Exception as e:
            print(f"TrueDepth: Failed to unregister {cls.__name__}: {e}")

    # Remove property from Scene
    try:
        del bpy.types.Scene.depthgenius
    except Exception as e:
        print(f"TrueDepth: Failed to remove scene property: {e}")

    print("TrueDepth: Unregistration complete")


if __name__ == "__main__":
    register()
