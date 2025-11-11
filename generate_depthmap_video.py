import shutil
import bpy
from bpy.props import StringProperty, IntProperty, FloatProperty, EnumProperty, BoolProperty
from bpy_extras.io_utils import ImportHelper
import os
from pathlib import Path
from .operators import checkpoint_exits

class DEPTHGENIUS_OT_GenerateVideoDepthMap(bpy.types.Operator):
    bl_idname = "depthgenius.generate_video_depth_map"
    bl_label = "Generate Video Depth Map"
    bl_description = "Generate depth map for a video file or movie clip"
    bl_options = {"REGISTER",}

    video = None
    out = None
    frame_count = 0
    current_frame = 0
    _timer = None
    output_folder = None

    @classmethod
    def poll(cls, context):
        depthgenius = context.scene.depthgenius
        return depthgenius.image and depthgenius.image.source == 'MOVIE' and not depthgenius.warning_message


    def clear_output_folder(self):
        print("clearnig folder...")
        print(self.output_folder)
        if self.output_folder and self.output_folder.exists():
            for item in self.output_folder.iterdir():
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)

    def modal(self, context, event):
        if event.type == 'ESC':
            self.cancel(context)
            return {'CANCELLED'}

        if event.type == 'TIMER':
            if self.current_frame >= self.frame_count:
                self.finish(context)
                return {'FINISHED'}
            print(f"processing frame {self.current_frame}")
            self.update_progress_ui(context)
            self.process_frame(context)
            self.current_frame += context.scene.depthgenius.frame_step
            self.report({'INFO'}, f"Processing frame {self.current_frame + 1} of {self.frame_count}")
            # Update progress
            # context.window_manager.progress_update(self.current_frame / self.frame_count)
            

        return {'PASS_THROUGH'}

    def execute(self, context):
        global cv2
        import cv2
        context.window.cursor_set('WAIT')

        depthgenius = context.scene.depthgenius
        pg = depthgenius.progress
        pg.is_running = True
        pg.progress = 0
        pg.status= ""

        self.filepath = bpy.path.abspath(depthgenius.image.filepath)
        print(self.filepath)
        self.video = cv2.VideoCapture(self.filepath)
        if not self.video.isOpened():
            pg.is_running = False
            self.report({'ERROR'}, "Error opening video file")
            context.window.cursor_set('DEFAULT')
            return {'CANCELLED'}

        # Get video properties
        self.fps = self.video.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))

        image = depthgenius.image
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
        
        if output_dir and depthgenius.save_as_video:
            output_dir.mkdir(parents=True, exist_ok=True)
            self.output_path = str(output_dir / f"depth_map_{str(img_filepath.stem)}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.out = cv2.VideoWriter(self.output_path, fourcc, self.fps, (self.width, self.height), isColor=True)
        elif output_dir and not depthgenius.save_as_video:
            output_dir = output_dir / f"{str(img_filepath.stem)}_depth_frames"
            output_dir.mkdir(parents=True, exist_ok=True)
            self.output_folder = output_dir
            self.clear_output_folder()
        else:
            self.output_path = None

        self._timer = context.window_manager.event_timer_add(0.1, window=context.window)
        context.window_manager.modal_handler_add(self)

        return {'RUNNING_MODAL'}

    def process_frame(self, context):
        depthgenius = context.scene.depthgenius
        self.video.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        ret, frame = self.video.read()
        if ret:
            from . import depth_estimation
            # Generate depth map for the frame
            depth_map = depth_estimation.main(
                depthgenius.model_size,
                checkpoint_exits(depthgenius.model_size)[1],
                frame,
                None,
                False,
                depthgenius.plane_removal_factor,
                depthgenius.use_colormap,
                depthgenius.colormap,
                include_alpha = depthgenius.include_alpha,
                save_16bit = True,
                preferred_device= depthgenius.device,
            )

            if depthgenius.save_as_video:
                # Write frame to video
                depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
                depth_map = depth_map.astype('uint8')
                self.out.write(depth_map)
            else:
                # Save frame as image
                frame_filename = f"depth_frame_{(self.current_frame+1):06d}.png"
                output_path = str(self.output_folder / frame_filename)
                cv2.imwrite(output_path, depth_map)

    def finish(self, context):
        depthgenius = context.scene.depthgenius
        self.video.release()
        if depthgenius.save_as_video:
            self.out.release()
            filepath = self.output_path
            output_message = f"Depth map video generated and saved as {self.output_path}"
        else:
            output_message = f"Depth map frames saved in folder: {self.output_folder}"
            frame_filename = f"depth_frame_{1:06d}.png"
            filepath = str(self.output_folder / frame_filename)
        depth_image = bpy.data.images.load(filepath, check_existing=True)
        if depthgenius.save_as_video:
            depth_image.source = 'MOVIE'
        else:
            depth_image.source = 'SEQUENCE'
        depth_image.reload()
        depthgenius.depth_map = depth_image
        depthgenius.depth_map.colorspace_settings.name = 'Non-Color'

        context.window_manager.event_timer_remove(self._timer)
        depthgenius.progress.is_running = False

        depthgenius.depth_map.update_tag()
        depthgenius.depth_map.preview_ensure()
        context.view_layer.update()
        context.area.tag_redraw()
        self.report({'INFO'}, output_message)
        context.window.cursor_set('DEFAULT')

    def cancel(self, context):
        if self.video:
            self.video.release()
        if self.out:
            self.out.release()
        context.window_manager.event_timer_remove(self._timer)
        context.scene.depthgenius.progress.is_running = False
        self.report({'INFO'}, "Video depth map generation cancelled")
        context.window.cursor_set('DEFAULT')

    def update_progress_ui(self, context):
        dg = context.scene.depthgenius
        pg = dg.progress
        last_frame_to_process = ((self.frame_count - 1) // dg.frame_step) * dg.frame_step
        total_frames_to_process = (last_frame_to_process // dg.frame_step) + 1
        current_processed_frame = (min(self.current_frame, last_frame_to_process) // dg.frame_step) + 1

        pg.progress = current_processed_frame / total_frames_to_process * 100
        pg.status = f"Processing frame {self.current_frame + 1} of {self.frame_count}"
        context.area.tag_redraw()

