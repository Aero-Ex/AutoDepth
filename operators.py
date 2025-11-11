import subprocess
import sys
import bpy
import os
from pathlib import Path
import numpy as np

# Global variables
CHECKPOINTS_DIR = "checkpoints"
MODEL_URLS = {
    "vits": "https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth",
    "vitb": "https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth",
    "vitl": "https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth",
    "vitg": "",
}

def get_addon_dir():
    return Path(bpy.path.abspath(__file__)).parent.resolve()

def checkpoint_exits(model_size):
    checkpoints_dir = get_addon_dir() / CHECKPOINTS_DIR
    checkpoints_dir.mkdir(exist_ok=True)
    
    url = MODEL_URLS.get(model_size)
    if not url:
        raise ValueError(f"Invalid model size: {model_size}")
    
    filename = f"depth_anything_v2_{model_size}.pth"
    filepath = checkpoints_dir / filename
    
    # if not filepath.exists():
    #     raise FileNotFoundError(f"Checkpoint for {model_size} not found: {filepath}")
    
    return filepath.exists(), filepath

def draw_func(self, context):
    row = self.layout.row()
    row.label(text = '')
class DEPTHGENIUS_OT_GenerateDepthMap(bpy.types.Operator):
    bl_idname = "depthgenius.generate_depth_map"
    bl_label = "Generate Depth Map"
    bl_description = "Generate depth map for the selected image"
    bl_options={"REGISTER","UNDO"}

    @classmethod
    def poll(cls, context):
        depthgenius = context.scene.depthgenius
        return depthgenius.image and not depthgenius.warning_message
    
    def execute(self, context):
        scene = context.scene
        depthgenius = scene.depthgenius
        context.window.cursor_set('WAIT')
        try:
            exists, checkpoint_path = checkpoint_exits(depthgenius.model_size)
            if not exists:
                self.error_msg = f"Checkpoint file for model {depthgenius.model_size} not found at location: {checkpoint_path}"
                self.report({'ERROR'}, self.error_msg)
                return {'CANCELLED'}
            # Prepare image
            image = depthgenius.image
            width, height = image.size
            
            img_filepath = Path(bpy.path.abspath(image.filepath))
            # Determine output path based on save_location
            if depthgenius.save_location == 'BLEND_FILE':
                output_dir = Path(bpy.path.abspath(bpy.data.filepath)).parent
            elif depthgenius.save_location == 'ORIGINAL_IMAGE':
                output_dir = img_filepath.parent if image.filepath else None
            elif depthgenius.save_location == 'CUSTOM_DIR':
                output_dir = Path(bpy.path.abspath(depthgenius.custom_save_dir)).resolve()
            else:  # 'NO_SAVE'
                output_dir = None

            # print(f"depth_map_{img_filepath.stem if image.filepath else image.name}.png")Z
            if output_dir:
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / f"depth_map_{str(img_filepath.stem) if image.filepath else image.name}.png"
            else:
                output_path = None
            
            from . import depth_estimation
            depth_map = depth_estimation.main(depthgenius.model_size,
                                              str(checkpoint_path),
                                              image,
                                              str(output_path),
                                              depthgenius.use_dirty_image,
                                              depthgenius.plane_removal_factor,
                                              depthgenius.use_colormap,
                                              depthgenius.colormap,
                                              include_alpha = depthgenius.include_alpha,
                                              save_16bit = True,
                                              preferred_device= depthgenius.device,)

            if output_path:
                # Load the saved depth map into Blender
                depth_image = bpy.data.images.load(str(output_path), check_existing=True)
                depth_image.reload()
            else:
                # Create a new image in Blender without saving to disk
                depth_image = bpy.data.images.get(f"Depth_Map_{image.name}")
                if depth_image is None:
                    depth_image = bpy.data.images.new(name=f"Depth_Map_{image.name}", width=width, height=height)
                
                depth_image = depth_estimation.cv2_to_blender_image(depth_map,depth_image)
                # depth_image.update_tag()
                # depth_image.pack()  # Pack the image data into the .blend file
            depthgenius.depth_map = depth_image
            if depthgenius.save_location != "NO_SAVE":
                depthgenius.depth_map.colorspace_settings.name = 'Non-Color'
            del depth_map
            depthgenius.depth_map.update_tag()
            depthgenius.depth_map.preview_ensure()
            context.view_layer.update()
            context.area.tag_redraw()

            self.report({'INFO'}, f"Depth map generated successfully and {f'saved @ {depthgenius.save_location}' if depthgenius.save_location != 'NO_SAVE' else F'stored in Blender as Image {depth_image.name}'}")

        except Exception as e:
            import traceback
            print(traceback.format_exc())
            self.report({'ERROR'}, f"Failed to generate depth map: {str(e)}")
            context.window.cursor_set('DEFAULT')
        context.window.cursor_set('DEFAULT')
        return {'FINISHED'}

class DEPTHGENIUS_OT_open_image_in_new_window(bpy.types.Operator):
    bl_idname = "depthgenius.open_in_new_window"
    bl_label = "Open Image in New Window"
    bl_description = "Opens the specified image in a new Image Editor window"
    bl_options = {'REGISTER',}

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

        return {'FINISHED'}


class DEPTHGENIUS_OT_open_checkpoint_folder(bpy.types.Operator):
    """Opens file explorer at the addon's checkpoint folder"""
    bl_idname = "depthgenius.open_checkpoint_folder"
    bl_label = "Open Checkpoint Folder"
    bl_description = "Opens the system file explorer at the addon's checkpoint folder"
    
    def execute(self, context):
        checkpoints_dir = get_addon_dir() / CHECKPOINTS_DIR
        
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
