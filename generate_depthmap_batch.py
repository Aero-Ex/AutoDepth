import math
import bpy
from bpy.props import CollectionProperty, StringProperty, BoolProperty, EnumProperty, FloatProperty
from bpy_extras.io_utils import ImportHelper
from pathlib import Path

from .operators import checkpoint_exits
from . import image_plane
def fast_grid_dimensions(n):
    rows = math.ceil(math.sqrt(n))
    cols = math.ceil(n / rows)
    return rows, cols

class DEPTHGENIUS_OT_BatchGenerateDepthMap(bpy.types.Operator, ImportHelper):
    bl_idname = "depthgenius.batch_generate_depth_map"
    bl_label = "Batch Generate Depth Maps"
    bl_description = "Generate depth maps for multiple images"
    bl_options = {"REGISTER",}

    files: CollectionProperty(
        name="File Path",
        type=bpy.types.OperatorFileListElement,
        options={'SKIP_SAVE'}
    )
    directory: StringProperty(subtype='DIR_PATH')

    filename_ext = ".png"
    filter_glob: StringProperty(default="*.png;*.jpg;*.jpeg", options={'HIDDEN'})


    generate_mesh: BoolProperty(
        name="Generate Mesh",
        description="Generate a 3D mesh from the depth map",
        default=True
    )

    mesh_arrangement: EnumProperty(
        name="Mesh Arrangement",
        items=[
            ('ROW', "Row", "Arrange meshes in a row"),
            ('GRID', "Grid", "Arrange meshes in a grid")
        ],
        default='GRID'
    )

    spacing: FloatProperty(
        name="Spacing",
        description="Space between meshes",
        default=0.5,
        min=0.0,
        soft_max=10.0
    )

    loop_cuts: bpy.props.IntProperty(default=10, soft_max=20, soft_min=0, min=0, max=100,options={'SKIP_SAVE'})
    
    execution_reached= False
    current_index = -1
    total_images = -1
    phase = ''
    _timer = None
    def modal(self, context,event):
        if event.type == 'ESC':
            print("esc: cancelling operator")
            self.cancel(context)
            return {'FINISHED'}

        if event.type == 'TIMER':
            context.window.cursor_set('WAIT')
            print(self.phase)
            if self.phase == "DEPTH_MAP":
                if self.current_index >= self.total_images:
                    self.phase = "MESH" if self.generate_mesh else "FINISH"
                    self.current_index = 0
                else:
                    self.update_progress_ui(context)
                    self.process_image(context)
                    self.current_index += 1
            
            elif self.phase == "MESH":
                if self.current_index >= self.total_images:
                    self.phase = "FINISH"
                else:
                    self.update_progress_ui(context)
                    self.process_mesh(context)
                    self.current_index += 1
            
            if self.phase == "FINISH":
                self.finish(context)
                return {'FINISHED'}

            context.window.cursor_set('DEFAULT')
        return {'PASS_THROUGH'}

    def update_progress_ui(self, context):
        depthgenius = context.scene.depthgenius
        pg = depthgenius.progress
        pg.progress = (self.current_index / self.total_images)*100
        if self.phase == "DEPTH_MAP":
            pg.status = f"Generating Depth Maps: {self.current_index + 1} / {self.total_images}"
        elif self.phase == "MESH":
            pg.status = f"Generating Meshes: {self.current_index + 1} / {self.total_images}"
    
    def finish(self, context):
        depthgenius = context.scene.depthgenius
        pg = depthgenius.progress
        self.update_progress_ui(context)
        context.view_layer.update()
        context.area.tag_redraw()
        context.window.cursor_set('DEFAULT')
        context.window_manager.event_timer_remove(self._timer)
        pg.is_running = False
        return {'FINISHED'}

    def cancel(self, context):
        if not self.execution_reached:
            return
        depthgenius = context.scene.depthgenius
        pg = depthgenius.progress
        self.update_progress_ui(context)
        pg.status = "Cancelled the Operator/break"+pg.status 
        context.view_layer.update()
        context.area.tag_redraw()
        context.window.cursor_set('DEFAULT')
        context.window_manager.event_timer_remove(self._timer)
        pg.is_running = False
        return {'FINISHED'}

    def process_image(self,context):
        try:
            scene = context.scene
            depthgenius = scene.depthgenius

            file_elem = self.files[self.current_index]
            filepath = Path(self.directory).resolve() / file_elem.name
            image = bpy.data.images.load(str(filepath), check_existing=True)
            image.reload()
            depthgenius.image = image
            width, height = image.size
            img_filepath = Path(bpy.path.abspath(image.filepath))

            #gop Determine output path based on save_location
            if depthgenius.save_location == 'BLEND_FILE':
                output_dir = Path(bpy.path.abspath(bpy.data.filepath)).parent
            elif depthgenius.save_location == 'ORIGINAL_IMAGE':
                output_dir = img_filepath.parent if image.filepath else None
            elif depthgenius.save_location == 'CUSTOM_DIR':
                output_dir = Path(bpy.path.abspath(depthgenius.custom_save_dir)).resolve()
            else:  # 'NO_SAVE'
                output_dir = None

            if output_dir:
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / f"depth_map_{str(img_filepath.stem) if image.filepath else image.name}.png"
            else:
                output_path = None
            #endgop
        
            from . import depth_estimation
            depth_map = depth_estimation.main(depthgenius.model_size,
                                            str(self.checkpoint_path),
                                            image,
                                            str(output_path),
                                            depthgenius.use_dirty_image,
                                            depthgenius.plane_removal_factor,
                                            depthgenius.use_colormap,
                                            depthgenius.colormap,
                                            include_alpha = depthgenius.include_alpha,
                                            save_16bit = True,
                                            preferred_device= depthgenius.device)
            if output_path:
                depth_image = bpy.data.images.load(str(output_path), check_existing=True)
                depth_image.reload()
            else: # Create a new image in Blender without saving to disk
                depth_image = bpy.data.images.get(f"Depth_Map_{image.name}")
                if depth_image is None:
                    depth_image = bpy.data.images.new(name=f"Depth_Map_{image.name}", width=width, height=height)
                depth_image = depth_estimation.cv2_to_blender_image(depth_map,depth_image)

            depthgenius.depth_map = depth_image
            depthgenius.depth_map.update_tag()
            # depthgenius.depth_map.preview_ensure()
            if depthgenius.save_location != "NO_SAVE":
                depthgenius.depth_map.colorspace_settings.name = 'Non-Color'
            
            # Store the image and depth_image for later mesh generation
            self.image_depth_pairs.append((image, depth_image))

            self.report({'INFO'}, f"Depth map generated successfully and {f'saved @ {depthgenius.save_location}' if depthgenius.save_location != 'NO_SAVE' else F'stored in Blender as Image {depth_image.name}'}")
            del depth_map
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            self.report({'ERROR'}, f"Failed to generate depth map: {str(e)}")
            context.window.cursor_set('DEFAULT')

    def process_mesh(self,context):
        image, depth_image = self.image_depth_pairs[self.current_index]
        width, height =  image_plane.compute_plane_size(image.size)
        plane =          image_plane.create_image_plane(context,"ImagePlane",width,height)
        _ =              image_plane.align_plane(plane,'Y-',height)
        _ =              self.position_mesh(plane)
        cuts_x, cuts_y = image_plane.calculate_cuts(self.loop_cuts,image.size[0], image.size[1])
        # print("X, Y: ",cuts_x,", ", cuts_y)
        _ =              image_plane.create_square_quads_with_loopcuts(plane, cuts_x, cuts_y)
        modifiers =      image_plane.add_modifiers(plane,image,depth_image)
        _ =              image_plane.add_crop_drivers(plane)
        mat = bpy.data.materials.new(name="TrueDepthMaterial")
        image_plane.create_material(mat,image, depth_image, modifiers[1].texture)

        plane.data.materials.append(mat)

    def position_mesh(self, mesh_obj):
        dimensions = mesh_obj.dimensions
        # Swap Y and Z dimensions to account for 90-degree rotation on X-axis
        width, height, depth = dimensions.x, dimensions.y, dimensions.z

        if self.mesh_arrangement == 'ROW':
            mesh_obj.location.x = self.current_offset_x
            self.current_offset_x += width + self.spacing
        elif self.mesh_arrangement == 'GRID':
            rows, cols = fast_grid_dimensions(self.total_images)
            row = self.current_index // cols
            col = self.current_index % cols
            mesh_obj.location.x = col * (self.max_width + self.spacing)
            mesh_obj.location.z = row * (self.max_height + self.spacing)  # Negative for Y axis

        # Update max dimensions for grid arrangement
        if self.mesh_arrangement == 'GRID':
            self.max_width = max(self.max_width, width)
            self.max_height = max(self.max_height, height)


    def execute(self, context):
        self.execution_reached = True
        scene = context.scene
        depthgenius = scene.depthgenius
        context.window.cursor_set('WAIT')

        self.total_images = len(self.files)
        self.current_index = 0
        self.phase = "DEPTH_MAP"
        self.image_depth_pairs = []

        # Initialize positioning variables
        self.current_offset_x = 0
        self.current_offset_z = 0
        self.max_width = 0
        self.max_height = 0

        pg = depthgenius.progress
        pg.progress = 0
        pg.status = f"Initializing..."
        pg.is_running= True
        
        exists, checkpoint_path = checkpoint_exits(depthgenius.model_size)
        self.checkpoint_path = checkpoint_path
        if not exists:
            self.error_msg = f"Checkpoint file for model {depthgenius.model_size} not found at location: {checkpoint_path}"
            pg.is_running = False
            self.report({'ERROR'}, self.error_msg)
            return {'CANCELLED'}
        # Prepare image
        self._timer = context.window_manager.event_timer_add(0.1, window=context.window)
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def draw(self, context):
        layout = self.layout
        row = layout.row()
        row.use_property_split = True
        row.use_property_decorate = False
        row.label(text="Generate Mesh")
        row.prop(self,"generate_mesh", toggle=True, text="YES" if self.generate_mesh else "NO")
        if self.generate_mesh:
            col = layout.column()
            col.use_property_split = True
            col.use_property_decorate = False   
            col.prop(self,"mesh_arrangement", expand=True)
            col.prop(self,"spacing",)
            col.prop(self,"loop_cuts")
        
        box = layout.box()
        l = len([file_elem for file_elem in self.files if not (Path(self.directory)/file_elem.name).is_dir()])
        box.label(text=f"Selected Images: {l}")
        # print(self.directory)
        for i, file_elem in enumerate(self.files):
            path:Path = Path(self.directory)/file_elem.name
            if path.is_dir():
                continue
            box.label(text=file_elem.name)

def register():
    bpy.utils.register_class(DEPTHGENIUS_OT_BatchGenerateDepthMap)

def unregister():
    bpy.utils.unregister_class(DEPTHGENIUS_OT_BatchGenerateDepthMap)