import subprocess
import sys
import bpy
from pathlib import Path

# Global variables
CHECKPOINTS_DIR = "checkpoints"
MODEL_URLS = {
    "vits": "https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth",
    "vitb": "https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth",
    "vitl": "https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth",
    "vitg": "https://huggingface.co/likeabruh/depth_anything_v2_vitg/resolve/main/depth_anything_v2_vitg.pth",
}


def get_addon_dir():
    """Get the addon directory path"""
    return Path(bpy.path.abspath(__file__)).parent.resolve()


def checkpoint_exits(model_size):
    """Check if checkpoint file exists for given model size"""
    checkpoints_dir = get_addon_dir() / CHECKPOINTS_DIR
    checkpoints_dir.mkdir(exist_ok=True)

    url = MODEL_URLS.get(model_size)
    if not url:
        raise ValueError(f"Invalid model size: {model_size}")

    filename = f"depth_anything_v2_{model_size}.pth"
    filepath = checkpoints_dir / filename

    return filepath.exists(), filepath


class DEPTHGENIUS_OT_GenerateDepthMap(bpy.types.Operator):
    """Generate depth map for the selected image"""
    bl_idname = "depthgenius.generate_depth_map"
    bl_label = "Generate Depth Map"
    bl_description = "Generate depth map for the selected image"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        depthgenius = context.scene.depthgenius
        return depthgenius.image and not depthgenius.warning_message

    def execute(self, context):
        scene = context.scene
        depthgenius = scene.depthgenius
        context.window.cursor_set('WAIT')

        try:
            # Check if checkpoint exists
            exists, checkpoint_path = checkpoint_exits(depthgenius.model_size)
            if not exists:
                error_msg = (f"Checkpoint file for model {depthgenius.model_size} not found at location: {checkpoint_path}\n"
                           f"Please download the model checkpoint first.")
                self.report({'ERROR'}, error_msg)
                context.window.cursor_set('DEFAULT')
                return {'CANCELLED'}

            # Add DLL directory for Windows BEFORE importing torch
            if sys.platform == "win32":
                from . import install_packages
                try:
                    install_packages.add_dll_directory(depthgenius.device)
                except Exception as e:
                    print(f"TrueDepth: Could not add DLL directory: {e}")

            # NOW import depth_estimation (which will import torch/cv2)
            print("TrueDepth: Importing depth_estimation module...")
            try:
                from . import depth_estimation
            except Exception as e:
                error_msg = f"Failed to import dependencies: {str(e)}"
                if "DLL" in str(e) and sys.platform == 'win32':
                    error_msg += "\n\nPlease install Visual C++ Redistributable:\nhttps://aka.ms/vs/17/release/vc_redist.x64.exe"
                self.report({'ERROR'}, error_msg)
                context.window.cursor_set('DEFAULT')
                return {'CANCELLED'}

            print("TrueDepth: Dependencies imported successfully!")

            # Prepare image
            image = depthgenius.image
            width, height = image.size

            img_filepath = Path(bpy.path.abspath(image.filepath)) if image.filepath else None

            # Determine output path based on save_location
            if depthgenius.save_location == 'BLEND_FILE':
                output_dir = Path(bpy.path.abspath(bpy.data.filepath)).parent
            elif depthgenius.save_location == 'ORIGINAL_IMAGE':
                output_dir = img_filepath.parent if img_filepath else None
            elif depthgenius.save_location == 'CUSTOM_DIR':
                output_dir = Path(bpy.path.abspath(depthgenius.custom_save_dir)).resolve()
            else:  # 'NO_SAVE'
                output_dir = None

            if output_dir:
                output_dir.mkdir(parents=True, exist_ok=True)
                output_filename = f"depth_map_{img_filepath.stem if img_filepath else image.name}.png"
                output_path = output_dir / output_filename
            else:
                output_path = None

            # Generate depth map
            print("TrueDepth: Generating depth map...")
            depth_map = depth_estimation.main(
                depthgenius.model_size,
                str(checkpoint_path),
                image,
                str(output_path) if output_path else None,
                depthgenius.use_dirty_image,
                depthgenius.plane_removal_factor,
                depthgenius.use_colormap,
                depthgenius.colormap,
                include_alpha=depthgenius.include_alpha,
                save_16bit=True,
                preferred_device=depthgenius.device,
                enable_cpu_offload=depthgenius.enable_cpu_offload,
            )

            if output_path:
                # Load the saved depth map into Blender
                depth_image = bpy.data.images.load(str(output_path), check_existing=True)
                depth_image.reload()
                depth_image.colorspace_settings.name = 'Non-Color'
            else:
                # Create a new image in Blender without saving to disk
                depth_image_name = f"Depth_Map_{image.name}"
                depth_image = bpy.data.images.get(depth_image_name)
                if depth_image is None:
                    depth_image = bpy.data.images.new(
                        name=depth_image_name,
                        width=width,
                        height=height
                    )

                depth_image = depth_estimation.cv2_to_blender_image(depth_map, depth_image)

            # Set the depth map property
            depthgenius.depth_map = depth_image
            del depth_map

            # Update UI
            depth_image.update_tag()
            depth_image.preview_ensure()
            context.view_layer.update()
            context.area.tag_redraw()

            # Report success
            if depthgenius.save_location != "NO_SAVE":
                self.report({'INFO'}, f"Depth map generated successfully and saved to {output_path}")
            else:
                self.report({'INFO'}, f"Depth map generated and stored in Blender as Image {depth_image.name}")

        except Exception as e:
            import traceback
            print("TrueDepth: Error generating depth map:")
            print(traceback.format_exc())
            error_msg = f"Failed to generate depth map: {str(e)}"
            if "DLL" in str(e) and sys.platform == 'win32':
                error_msg += "\n\nDLL Error - Please install Visual C++ Redistributable:\nhttps://aka.ms/vs/17/release/vc_redist.x64.exe"
            self.report({'ERROR'}, error_msg)
        finally:
            context.window.cursor_set('DEFAULT')

        return {'FINISHED'}


class DEPTHGENIUS_OT_open_image_in_new_window(bpy.types.Operator):
    """Opens the specified image in a new Image Editor window"""
    bl_idname = "depthgenius.open_in_new_window"
    bl_label = "Open Image in New Window"
    bl_description = "Opens the specified image in a new Image Editor window"
    bl_options = {'REGISTER'}

    image_name: bpy.props.StringProperty(
        name="Image Name",
        description="Name of the image to open in the new window",
        default=""
    )

    def execute(self, context):
        # Check if the image exists
        image = bpy.data.images.get(self.image_name)
        if not image:
            self.report({'ERROR'}, f"Image '{self.image_name}' not found")
            return {'CANCELLED'}

        # Create a new window
        bpy.ops.wm.window_new()

        # Get the newly created window and its area
        window = bpy.context.window_manager.windows[-1]
        area = window.screen.areas[0]

        # Change the area type to IMAGE_EDITOR
        area.type = 'IMAGE_EDITOR'

        # Set the image in the editor
        area.spaces.active.image = image

        self.report({'INFO'}, f"Opened {self.image_name} in new window")
        return {'FINISHED'}


class DEPTHGENIUS_OT_open_checkpoint_folder(bpy.types.Operator):
    """Opens file explorer at the addon's checkpoint folder"""
    bl_idname = "depthgenius.open_checkpoint_folder"
    bl_label = "Open Checkpoint Folder"
    bl_description = "Opens the system file explorer at the addon's checkpoint folder"

    def execute(self, context):
        checkpoints_dir = get_addon_dir() / CHECKPOINTS_DIR
        checkpoints_dir.mkdir(exist_ok=True)

        try:
            # Open file explorer based on operating system
            if sys.platform == 'win32':  # Windows
                subprocess.Popen(['explorer', str(checkpoints_dir)])
            elif sys.platform == 'darwin':  # macOS
                subprocess.Popen(['open', str(checkpoints_dir)])
            else:  # Linux and other Unix
                subprocess.Popen(['xdg-open', str(checkpoints_dir)])

            self.report({'INFO'}, f"Opened checkpoint folder: {checkpoints_dir}")
            return {'FINISHED'}

        except Exception as e:
            self.report({'ERROR'}, f"Failed to open folder: {str(e)}")
            return {'CANCELLED'}
