import bpy
import sys
from pathlib import Path
from . import install_packages

pythonversion = f"python{sys.version_info.major}.{sys.version_info.minor}"


class DEPTHGENIUS_Preferences(bpy.types.AddonPreferences):
    """Addon preferences for TrueDepth"""
    bl_idname = __package__

    dependencies_path: bpy.props.StringProperty(
        name="Install path",
        description="Directory where additional dependencies for the addon are downloaded (NEEDS ~8GB SPACE)",
        subtype='DIR_PATH',
        default=str(Path("~/TrueDepth-blender-addon/").expanduser().resolve() / pythonversion)
    )

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "dependencies_path")

        box = layout.box()
        install_packages.draw(box, context.scene.depthgenius)

        layout.row().operator("depthgenius.open_checkpoint_folder")
        layout.row().prop(
            context.scene.depthgenius.progress,
            "is_running",
            text="Unlock UI",
            invert_checkbox=True
        )
