from pathlib import Path
import bpy
from . import install_packages

def get_install_folder(internal_folder):
    return Path(bpy.context.preferences.addons[__package__].preferences.dependencies_path) / internal_folder

def draw_color_palettes(layout, depthgenius):
    row = layout.row(align=True)
    row.popover("DEPTHGENIUS_PT_HoverInfo",icon='ERROR',text="")
    row.prop(depthgenius, "use_colormap",toggle=True, text="Use Color Palette")
    subrow = row.row()
    subrow.active = depthgenius.use_colormap
    subrow.prop(depthgenius,"colormap",text="")

def find_bump_node(obj):
    if not obj or not obj.active_material or not obj.active_material.use_nodes:
        return None
    
    material = obj.active_material
    nodes = material.node_tree.nodes
    output_node = next((node for node in nodes if node.type == 'OUTPUT_MATERIAL'), None)
    
    if not output_node or not output_node.inputs['Surface'].is_linked:
        return None
        
    surface_node = output_node.inputs['Surface'].links[0].from_node
    if surface_node.type != 'BSDF_PRINCIPLED':
        return None
        
    normal_input = surface_node.inputs['Normal']
    if not normal_input.is_linked:
        return None
        
    normal_node = normal_input.links[0].from_node
    return normal_node if normal_node.type == 'BUMP' else None

def find_principled_node(obj):
    if not obj or not obj.active_material or not obj.active_material.use_nodes:
        return None
    
    material = obj.active_material
    nodes = material.node_tree.nodes
    output_node = next((node for node in nodes if node.type == 'OUTPUT_MATERIAL'), None)
    
    if not output_node or not output_node.inputs['Surface'].is_linked:
        return None
        
    surface_node = output_node.inputs['Surface'].links[0].from_node
    if surface_node.type != 'BSDF_PRINCIPLED':
        return None
    
    if surface_node.inputs['Roughness'].is_linked:
        None
    
    return surface_node
class DEPTHGENIUS_PT_Panel(bpy.types.Panel):
    bl_label = "TrueDepth"
    bl_idname = "DEPTHGENIUS_PT_Panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'TrueDepth'
    
    def draw(self, context):
        layout = self.layout
        scene = context.scene
        depthgenius = scene.depthgenius
        
        continue_draw = install_packages.draw(layout,depthgenius)
        if not continue_draw:
            return

        layout.enabled = not depthgenius.progress.is_running
        col = layout.column()
        col.use_property_split = True
        col.use_property_decorate = False
        col.prop(depthgenius, "model_size")
        if not depthgenius.current_model_avaibility:
            box = layout.box()
            box.label(text=f"Selected Model {depthgenius.model_size} is not available.\nYou can download it from", icon='ERROR')
            op = box.operator("wm.url_open", text="Download it from here", icon="URL")
            op.url = "https://github.com/DepthAnything/Depth-Anything-V2?tab=readme-ov-file#pre-trained-models"
            return
        flag = 'Yes' if depthgenius.use_dirty_image else 'No'
        col.prop(depthgenius, "use_dirty_image", toggle=True, text=f"Use Dirty Image: {flag}")
        col.prop(depthgenius, "save_location")
        if depthgenius.save_location == 'CUSTOM_DIR':
            col.prop(depthgenius, "custom_save_dir")
        
        if depthgenius.warning_message:
            col.alert = True
            col.label(text=depthgenius.warning_message, icon='ERROR')

        box = layout.box()
        row = box.row()
        row.label(text="Mode: ")
        row.prop(depthgenius,"mode",expand=True)
        col = box.column()
        col.prop(depthgenius,"include_alpha", toggle=True)
        
        box = layout.box()
        if depthgenius.mode in ('IMAGE','MOVIE'):
            if depthgenius.mode == 'IMAGE':
                label = "Select Image for depth generation"
                operator = "depthgenius.generate_depth_map"
            if depthgenius.mode == 'MOVIE':
                label = "Select Video for depth generation"
                operator = "depthgenius.generate_video_depth_map"
            
            box.label(text= label, icon="TRIA_RIGHT")

            draw_color_palettes(box,depthgenius)
            
            if depthgenius.image:
                box.template_ID_preview(depthgenius, "image",open="image.open", rows = 3, cols = 5)
            else:
                box.template_ID(depthgenius, "image",open="image.open",)
            col = box.column()
            col.use_property_decorate = False
            col.use_property_split = True
            col.prop(depthgenius,"plane_removal_factor")
            if depthgenius.mode == 'MOVIE':
                col.prop(depthgenius,"save_as_video",)
                col.prop(depthgenius,"frame_step")
            
            col = box.column()
            col.scale_y = 2.0
            col.operator(operator)
        if depthgenius.mode == 'BATCH_IMG':
            label = "Select multiple images for depth generation"
            box.label(text= label, icon="TRIA_RIGHT")
            draw_color_palettes(box, depthgenius)
            col = box.column()
            col.use_property_decorate = False
            col.use_property_split = True
            col.prop(depthgenius,"plane_removal_factor")
            col = box.column()
            col.scale_y = 2.0
            col.operator("depthgenius.batch_generate_depth_map")
        if depthgenius.progress.is_running == True or depthgenius.progress.status != '':
            box = box.box()
            box.enabled = False
            for status in depthgenius.progress.status.split("/break"):
                box.label(text=status)
            box.prop(depthgenius.progress,"progress", slider=True)
        col.separator()

        if depthgenius.mode in ('IMAGE','MOVIE'):
            box = layout.box()
            if depthgenius.depth_map:
                row = box.row()
                row.label(text="Depth Image:", icon="TRIA_RIGHT")
                box.template_ID_preview(depthgenius, "depth_map",rows = 3, cols = 5)
                op = row.operator('depthgenius.open_in_new_window', text="View Generated Image")
                op.image_name = depthgenius.depth_map.name
            else:
                box.label(text="Select generated depth image", icon="TRIA_RIGHT")
                box.template_ID_preview(depthgenius, "depth_map",open="image.open", rows = 3, cols = 5)
                # layout.box().template_icon(bpy.types.UILayout.icon(depthgenius.depth_map), scale = 7)

            col = box.column()
            col.scale_y = 2.0
            col.operator("depthgenius.createplane", text="Create Mesh", icon="OUTLINER_OB_MESH")

class DEPTHGENIUS_PT_Modifiers(bpy.types.Panel):
    bl_label = "TrueDepth Modifier Properties"
    bl_idname = "DEPTHGENIUS_PT_Modifiers"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'TrueDepth'

    def draw_modifier_title(self, layout, modifier, icon):
        row = layout.row()
        subrow = row.row()
        subrow.label(text=modifier.name, icon=icon)
        subrow = row.row()
        subrow.prop(modifier, "show_viewport", icon_only=True)

    def draw_displacement_properties(self,layout, modifier):
        box = layout.box()
        self.draw_modifier_title(box,modifier,"MOD_DISPLACE")
        # Displacement strength
        row = box.row(align=True)
        row.use_property_decorate = False
        row.use_property_split = True
        row.prop(modifier, "strength", text="Strength")
        
        # Texture properties
        if modifier.texture:
            tex = modifier.texture
            box.label(text="Texture: " + tex.name, icon = "TEXTURE")
            box.use_property_decorate = False
            box.use_property_split = True
            # Crop properties
            col = box.column(align=True)
            # col.prop(tex, "crop_rectangle")
            col.prop(tex, "crop_min_x", text="Minimum X")
            col.prop(tex, "crop_min_y", text="Y")

            col = box.column(align=True)
            col.prop(tex, "crop_max_x", text="Maximum X")
            col.prop(tex, "crop_max_y", text="Y")
    
    def draw(self, context):
        layout = self.layout
        scene = context.scene

        active_obj = context.active_object
        if active_obj:
            cf = layout.grid_flow(row_major=True, columns=0, even_columns=True, even_rows=False, align=False)
            for mod in active_obj.modifiers:
                if mod.type == "SUBSURF":
                    box = cf.box()
                    self.draw_modifier_title(box,mod,"MOD_SUBSURF")
                    box.use_property_split = True
                    box.use_property_decorate = False
                    col = box.column(align= True)
                    col.prop(mod,"levels", text= "Levels Viewport")
                    col.prop(mod,"render_levels", text="Render")

                if mod.name == "TD_Displace_Depth" or mod.name == "TD_Displace_Detail(color)":
                    self.draw_displacement_properties(cf,mod)

                if bpy.app.version >= (3,1,0) and mod.name == "TD_delete_geo":
                        box = layout.box()
                        self.draw_modifier_title(box,mod,"GEOMETRY_NODES")
                        box.use_property_split = True
                        box.use_property_decorate = False
                        box.prop(mod,'["Input_6"]', text="Delete Geometry", toggle=True)
                        if mod["Input_6"]:
                            row = box.row(align=True)
                            row.prop(mod,'["Input_5"]', text="Distance Based", toggle=True)
                            row.prop(mod,'["Input_5"]', text="Boolean mesh", toggle=True, invert_checkbox=True)

                            if mod["Input_5"]:
                                box.label(text="Distance from")
                                box.prop(mod,'["Input_2"]', text = "Back")
                                box.prop(mod,'["Input_3"]', text="Front")
                            else:
                                box.prop_search(mod,'["Input_4"]',bpy.data,"objects", icon="OBJECT_DATA")
                
                if bpy.app.version >= (3,5,0) and mod.name == "TD_smooth_boundary":
                    box = layout.box()
                    self.draw_modifier_title(box,mod,"GEOMETRY_NODES")
                    box.use_property_split = True
                    box.use_property_decorate = False
                    box.prop(mod,'["Input_2"]', text = "Relax iteration")
                    box.prop(mod,'["Input_3"]', text="Relax Strength")
                    box.prop(mod,'["Input_4"]', text="Increase Boundary Mask")

                if mod.name == "TD_Smooth":
                    box = layout.box()
                    self.draw_modifier_title(box,mod,"MOD_SMOOTH")
                    box.use_property_split = True
                    box.use_property_decorate = False
                    box.prop(mod,"iterations")
                    box.prop(mod,"factor")

                if bpy.app.version >= (3,1,0) and mod.name == "TD_Base":
                        box = layout.box()
                        self.draw_modifier_title(box,mod,"GEOMETRY_NODES")
                        box.use_property_split = True
                        box.use_property_decorate = False
                        box.prop(mod,'["Input_2"]', text="Add Base", toggle=True)
                        if mod["Input_2"]:
                            box.prop(mod,'["Socket_2"]', text="Base Height")
                            box.prop(mod,'["Input_0"]', text = "Fill Cap")
                            box.prop(mod,'["Input_1"]', text="Smooth Shade Base")
                            box.prop_search(mod,'["Input_3"]',bpy.data,"materials", icon="MATERIAL")
                
            if bpy.app.version < (3,1,0):
                box = layout.box()
                box.label(text="'Delete Geometry' & 'Extrude Base' feature is available in Blender versions 3.1 or newer")
            if bpy.app.version < (3,5,0):
                box = layout.box()
                box.label(text="'Smooth Boundary' feature is available in Blender versions 3.5 or newer")
            if "TD_Displace_Depth" not in active_obj.modifiers:
                box = layout.box()
                box.label(text="Not Found: Displacment modifier with name TD_Displace_Depth")
            if "TD_Displace_Detail(color)" not in active_obj.modifiers:
                box = layout.box()
                box.label(text="Not Found: Displacment modifier with name TD_Displace_Detail(color)")
            if "TD_Smooth" not in active_obj.modifiers:
                box = layout.box()
                box.label(text="Not Found: Smooth modifier with name TD_Smooth")
        
            bump_node = find_bump_node(active_obj)
            if bump_node is not None:
                box = layout.box().column(align=True)
                box.label(text="Material: Bump", icon="MATERIAL")
                box.separator()
                box.prop(bump_node.inputs[0],"default_value", slider=True, text="Strength")
                box.prop(bump_node.inputs[1],"default_value", slider=True, text="Distance")
            principled = find_principled_node(active_obj)
            if principled is not None:
                box = layout.box().column(align=True)
                box.label(text="Material: Roughness", icon="MATERIAL")
                box.separator()
                box.prop(principled.inputs["Roughness"],"default_value", slider=True, text="Roughness")
        else:
            box = layout.box()
            box.label(text="Not Found: Active object")

class DEPTHGENIUS_PT_HoverInfo(bpy.types.Panel):
    bl_label = "TrueDepth Info"
    bl_idname = "DEPTHGENIUS_PT_HoverInfo"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'TrueDepth'
    bl_options = {'INSTANCED'}
    bl_description="Using Colorpalette with depthmap is only for asethic and visual purpose\
        Displacment may not work properly and can give weird results if color palette is used"

    def draw(self, context):
        layout = self.layout
        layout.label(text="Using Colorpalette with depthmap is only for asethic and visual purpose.")
        layout.label(text="Displacment may not work properly and can give weird results if color palette is used")

class DEPTHGENIUS_PT_Info(bpy.types.Panel):
    bl_idname = "DEPTHGENIUS_PT_Info"
    bl_label = "TrueDepth Links"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = 'TrueDepth'
    bl_order = 120

    urls = {
        'documentation' : 'https://blendermarket.com/products/truedepth/docs',
        'discord' : 'https://discord.com/invite/B3Ux4sxAzT',
        'review' : 'https://blendermarket.com/products/truedepth/ratings'
    }

    def draw(self, context):
        layout = self.layout
        layout.operator("wm.url_open", text='Documentation', icon='HELP').url = self.urls['documentation']
        layout.operator("wm.url_open", text='Community', icon='URL').url = self.urls['discord']
        layout.operator("wm.url_open", text='Enjoying? Leave a Review', icon='FUND').url = self.urls['review']
