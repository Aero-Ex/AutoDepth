from collections import namedtuple
from pathlib import Path
import bpy
import bmesh
ImageSpec = namedtuple(
    'ImageSpec',
    ['image', 'size','orient'])
     
class DEPTHGENIUS_OT_CreatePlane(bpy.types.Operator):
    bl_idname = "depthgenius.createplane"
    bl_label = "Create Plane"
    bl_description = "Create Mesh"
    bl_options = {"REGISTER", "UNDO"}

    axis: bpy.props.EnumProperty(
        name="Axis",
        description="Choose an axis",
        items=[
            ('X+', "X+", "Positive X axis"),
            ('Y+', "Y+", "Positive Y axis"),
            ('Z+', "Z+", "Positive Z axis"),
            ('X-', "X-", "Negative X axis"),
            ('Y-', "Y-", "Negative Y axis"),
            ('Z-', "Z-", "Negative Z axis"),
        ],
        default='Y-'
    )

    loop_cuts: bpy.props.IntProperty(default=10, soft_max=20, soft_min=0, min=0, max=100,options={'SKIP_SAVE'})

    @classmethod
    def poll(cls, context):
        return True

    def execute(self, context):
        depthgenius = context.scene.depthgenius
        image = depthgenius.image
        
        if not depthgenius.depth_map:
            self.report({'ERROR'}, "Please generate a depth map first")
            return {'CANCELLED'}
        
        if not depthgenius.image:
            self.report({'ERROR'}, "Please select the image first")
            return {'CANCELLED'}
        
        width, height =  compute_plane_size(image.size)
        plane =          create_image_plane(context,"ImagePlane",width,height)
        _ =              align_plane(plane,self.axis,height)
        cuts_x, cuts_y = calculate_cuts(self.loop_cuts,image.size[0], image.size[1])
        # print("X, Y: ",cuts_x,", ", cuts_y)
        _ =              create_square_quads_with_loopcuts(plane, cuts_x, cuts_y)
        modifiers =      add_modifiers(plane,image, depthgenius.depth_map)
        _ =              add_crop_drivers(plane)
        mat = bpy.data.materials.new(name="TrueDepthMaterial")
        create_material(mat,image, depthgenius.depth_map, modifiers[1].texture)

        plane.data.materials.append(mat)

        return {"FINISHED"}

def get_nodetree(tree_name:str, file_name:str) -> bpy.types.NodeTree:
    """
    Ensure the NodeTree 'TrueDepth_create_base_3.1' is available
    in the current blend file.  
    • If it already exists, just return it.  
    • Otherwise append it from  ./assets/TrueDepth_create_base_3.1.blend
      (relative to this Python file) and return the freshly-imported tree.
    """
    if tree_name in bpy.data.node_groups:
        return bpy.data.node_groups[tree_name]

    script_dir = Path(__file__).resolve().parent
    blend_path = script_dir / "assets" / f"{file_name}.blend"
    if not blend_path.is_file():
        raise FileNotFoundError(
            f"Couldn’t locate '{blend_path}'. "
            "Make sure the blend file sits in an 'assets' folder next to this script."
        )

    with bpy.data.libraries.load(str(blend_path), link=False) as (src, dst):
        if tree_name not in src.node_groups:
            raise ValueError(
                f"NodeTree '{tree_name}' isn’t present in {blend_path}."
            )
        dst.node_groups.append(tree_name)

    return bpy.data.node_groups[tree_name]

def add_modifiers(plane, image, depth_map):
    modifiers = []
    image = image
    subd = plane.modifiers.new('Subdivision', 'SUBSURF')
    subd.subdivision_type = 'SIMPLE'
    subd.levels = 3
    modifiers.append(subd)

    disp_depth = plane.modifiers.new('TD_Displace_Depth', 'DISPLACE')
    disp_depth.texture_coords = 'UV'
    disp_depth.strength = 0.6
    texture = bpy.data.textures.new(f'TD_Depth_{image.name}','IMAGE')
    texture.image = depth_map
    texture.use_calculate_alpha = True
    texture.extension = 'EXTEND'
    disp_depth.texture = texture
    modifiers.append(disp_depth)

    disp_detail = plane.modifiers.new('TD_Displace_Detail(color)', 'DISPLACE')
    disp_detail.texture_coords = 'UV'
    disp_detail.direction = 'Z'
    disp_detail.space = 'LOCAL'
    disp_detail.strength = 0.25
    texture = bpy.data.textures.new(f'TD_{image.name}','IMAGE')
    texture.image = image
    texture.use_calculate_alpha = True
    texture.extension = 'EXTEND'
    texture.contrast = 0.1
    disp_detail.texture = texture
    modifiers.append(disp_detail)

    if bpy.app.version >= (3,1,0):
        geonode = plane.modifiers.new('TD_delete_geo', 'NODES')
        nodetree = get_nodetree("TrueDepth_remove_geometry_3.1","TrueDepth_create_base_3.1")
        geonode.node_group = nodetree
        modifiers.append(geonode)

    if bpy.app.version >= (3,5,0):
        geonode = plane.modifiers.new('TD_smooth_boundary', 'NODES')
        nodetree = get_nodetree("TrueDepth_smooth_boundary_3.5","TrueDepth_smooth_boundary_3.5")
        geonode.node_group = nodetree
        modifiers.append(geonode)

    smooth = plane.modifiers.new('TD_Smooth', 'SMOOTH')
    smooth.factor = 0.1
    smooth.iterations = 20
    modifiers.append(smooth)

    if bpy.app.version >= (3,1,0):
        geonode = plane.modifiers.new('TD_Base', 'NODES')
        nodetree = get_nodetree("TrueDepth_create_base_3.1","TrueDepth_create_base_3.1")
        geonode.node_group = nodetree
        modifiers.append(geonode)
    return modifiers

def add_crop_drivers(obj):
    depth_mod = obj.modifiers.get("TD_Displace_Depth")
    detail_mod = obj.modifiers.get("TD_Displace_Detail(color)")
    
    if not depth_mod or not detail_mod:
        print("Required modifiers not found.")
        return
    
    depth_tex = depth_mod.texture
    detail_tex = detail_mod.texture
    
    if not depth_tex or not detail_tex:
        print("Textures not found in modifiers.")
        return
    
    crop_properties = ["crop_min_x", "crop_max_x", "crop_min_y", "crop_max_y"]
    
    for prop in crop_properties:
        if hasattr(detail_tex, prop) and hasattr(depth_tex, prop):
            driver = detail_tex.driver_add(prop).driver
            driver.type = 'AVERAGE'
            
            var = driver.variables.new()
            var.name = prop
            var.type = 'SINGLE_PROP'
            
            target = var.targets[0]
            target.id_type = 'TEXTURE'
            target.id = depth_tex
            target.data_path = prop
        else:
            print(f"Property {prop} not found in one or both textures.")

def create_image_plane(context, name, width, height):
    # Create new mesh
    bpy.ops.mesh.primitive_plane_add('INVOKE_REGION_WIN')
    plane = context.active_object
    # Why does mesh.primitive_plane_add leave the object in edit mode???
    if plane.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    plane.dimensions = width, height, 0.0
    plane.data.name = plane.name = name
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

    return plane

def compute_plane_size(size, height = 1.0):
    px, py = size
    # can't load data
    if px == 0 or py == 0:
        px = py = 1

    y = height
    x = px / py * y
    return x, y

from mathutils import Vector
from math import pi, ceil

axis_id_to_vector = {
    'X+': Vector(( 1,  0,  0)),
    'Y+': Vector(( 0,  1,  0)),
    'Z+': Vector(( 0,  0,  1)),
    'X-': Vector((-1,  0,  0)),
    'Y-': Vector(( 0, -1,  0)),
    'Z-': Vector(( 0,  0, -1)),
}

def align_plane(plane, axis,size_y):

    axis = axis_id_to_vector[axis] # front facing
    # rotate accordingly for x/y axiis
    if not axis.z:
        plane.rotation_euler.x = pi / 2

        if axis.y > 0:
            plane.rotation_euler.z = pi
        elif axis.y < 0:
            plane.rotation_euler.z = 0
        elif axis.x > 0:
            plane.rotation_euler.z = pi / 2
        elif axis.x < 0:
            plane.rotation_euler.z = -pi / 2

        plane.location.z = size_y/2
    # or flip 180 degrees for negative z
    elif axis.z < 0:
        plane.rotation_euler.y = pi

def calculate_cuts(loop_cuts, width_px, height_px, per_pixel = False):
    # Determine which dimension is shorter and calculate cuts
    if per_pixel:
        cuts_x = width_px - 1
        cuts_y = height_px - 1
        return cuts_x, cuts_y
    
    dim_y = 1.0
    dim_x = width_px/height_px
    # print("dim_X, dim_y: ",dim_x,", ", dim_y)
    if dim_x < dim_y:
        cuts_x = loop_cuts
        cuts_y = ceil(cuts_x * dim_y / dim_x)
    else:
        cuts_y = loop_cuts
        cuts_x = ceil(cuts_y * dim_x / dim_y)
    return cuts_x, cuts_y

def edge_loops(bm,edge):
    def walk(edge):
        yield edge
        edge.tag = True
        for l in edge.link_loops:
            loop = l.link_loop_radial_next.link_loop_next.link_loop_next
            if not (len(loop.face.verts) != 4 or loop.edge.tag):
                yield from walk(loop.edge)
    for e in bm.edges:
        e.tag = False
    return list(walk(edge))

def create_square_quads_with_loopcuts(obj, cuts_x, cuts_y,):
   # Get the mesh
    mesh = obj.data
    
    # Create a bmesh from the mesh
    bm = bmesh.new()
    bm.from_mesh(mesh)
    
    # Ensure lookup table is initialized
    bm.faces.ensure_lookup_table()
    
    # Get the single face
    face = bm.faces[0]
    face.smooth = True
    # Perform grid fill
    if cuts_x > 0 and cuts_y > 0:
        result = bmesh.ops.subdivide_edgering(
            bm,
            edges=[face.edges[0],face.edges[2]],
            cuts=cuts_x,
            profile_shape='INVERSE_SQUARE',
            profile_shape_factor=0.0,
        )
        
#        # Get the new edges created by the subdivision
#        new_edges = [e for e in result['geom_inner'] if isinstance(e, bmesh.types.BMEdge)]
#        
#        # Subdivide the edges to create the final grid
    bmesh.ops.subdivide_edgering(
        bm,
        edges=edge_loops(bm,face.edges[0]),
        cuts=cuts_y,
        profile_shape='INVERSE_SQUARE',
        profile_shape_factor=0.0,
    )    
    # Update the mesh
    bm.to_mesh(mesh)
    bm.free()
    
    # Update the mesh to show changes
    mesh.update()

    return obj

def create_material(mat,image, depth_map, driver_texture):
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    
    # Clear existing nodes
    nodes.clear()
    
    # Create nodes
    node_tex_coord = nodes.new(type='ShaderNodeTexCoord')
    node_mapping = nodes.new(type='ShaderNodeMapping')
    node_tex_image = nodes.new(type='ShaderNodeTexImage')
    node_disp_tex_image = nodes.new(type='ShaderNodeTexImage')
    node_displacement = nodes.new(type='ShaderNodeDisplacement')
    node_principled_bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
    node_material_output = nodes.new(type='ShaderNodeOutputMaterial')
    
    node_bump = nodes.new(type='ShaderNodeBump')
    combine_xyz_1 = nodes.new(type='ShaderNodeCombineXYZ')
    combine_xyz_2 = nodes.new(type='ShaderNodeCombineXYZ')
    subtract_1 = nodes.new(type='ShaderNodeMath')
    subtract_2 = nodes.new(type='ShaderNodeMath')
    MinX = nodes.new(type='ShaderNodeValue')
    MinY = nodes.new(type='ShaderNodeValue')
    MaxX = nodes.new(type='ShaderNodeValue')
    MaxY = nodes.new(type='ShaderNodeValue')
    

    # Set up nodes
    node_principled_bsdf.inputs["Roughness"].default_value = 0.8
    node_bump.inputs[0].default_value = 0.2
    node_bump.inputs[1].default_value = 1.0

    node_tex_image.image = image
    node_tex_image.image_user.use_auto_refresh = True
    node_tex_image.image_user.frame_duration = image.frame_duration
    node_disp_tex_image.image = depth_map
    node_disp_tex_image.image_user.use_auto_refresh = True
    node_disp_tex_image.image_user.frame_duration = depth_map.frame_duration
    node_displacement.inputs[2].default_value = 0.0
    
    subtract_1.operation = 'SUBTRACT'
    subtract_1.hide = True
    subtract_2.operation = 'SUBTRACT'
    subtract_2.hide = True
    
    combine_xyz_1.hide = True
    combine_xyz_2.hide = True
    MinX.label = "Min_X"
    MinY.label = "Min_Y"
    MaxX.label = "Max_X"
    MaxY.label = "Max_Y"
    
    def add_driver_to_value_node(node,driver_texture,prop):
        driver = node.outputs[0].driver_add("default_value").driver
        driver.type = 'AVERAGE'
        # Create a new variable for the driver
        var = driver.variables.new()
        var.name = prop
        var.type = 'SINGLE_PROP'
        # Set the variable's target
        target = var.targets[0]
        target.id_type = 'TEXTURE'
        target.id = driver_texture
        target.data_path = prop
    
    add_driver_to_value_node(MinX,driver_texture,"crop_min_x")
    add_driver_to_value_node(MinY,driver_texture,"crop_min_y")
    add_driver_to_value_node(MaxX,driver_texture,"crop_max_x")
    add_driver_to_value_node(MaxY,driver_texture,"crop_max_y")
    # Link nodes
    links.new(MaxX.outputs['Value'], subtract_1.inputs[0])
    links.new(MinX.outputs['Value'], subtract_1.inputs[1])

    links.new(MaxY.outputs['Value'], subtract_2.inputs[0])
    links.new(MinY.outputs['Value'], subtract_2.inputs[1])

    links.new(subtract_1.outputs['Value'], combine_xyz_1.inputs['X'])
    links.new(subtract_2.outputs['Value'], combine_xyz_1.inputs['Y'])

    links.new(MinX.outputs['Value'], combine_xyz_2.inputs['X'])
    links.new(MinY.outputs['Value'], combine_xyz_2.inputs['Y'])

    links.new(combine_xyz_2.outputs['Vector'], node_mapping.inputs['Location'])
    links.new(combine_xyz_1.outputs['Vector'], node_mapping.inputs['Scale'])

    links.new(node_tex_coord.outputs['UV'], node_mapping.inputs['Vector'])
    links.new(node_mapping.outputs['Vector'], node_tex_image.inputs['Vector'])
    links.new(node_mapping.outputs['Vector'], node_disp_tex_image.inputs['Vector'])
    links.new(node_disp_tex_image.outputs['Color'], node_displacement.inputs['Height'])
    links.new(node_displacement.outputs['Displacement'], node_material_output.inputs['Displacement'])
    links.new(node_tex_image.outputs['Color'],node_principled_bsdf.inputs['Base Color'])
    links.new(node_tex_image.outputs['Color'],node_bump.inputs['Height'])
    links.new(node_bump.outputs['Normal'],node_principled_bsdf.inputs['Normal'])
    links.new(node_principled_bsdf.outputs['BSDF'],node_material_output.inputs['Surface'])

    node_material_output.location = (1050,106)
    node_displacement.location = (829, -92)
    node_principled_bsdf.location = (571, 225)

    node_bump.location = (290, -298)
    offset = 400
    node_tex_image.location = (257-offset, 215)
    node_disp_tex_image.location = (262-offset, -131)
    node_mapping.location = (15-offset, -149)
    node_tex_coord.location = (-243-offset, 146)
    combine_xyz_2.location = (-285-offset, -180)
    combine_xyz_1.location = (-204-offset, -315)
    subtract_1.location = (-432-offset, -283)
    subtract_2.location = (-428-offset, -354)
    MinX.location = (-732-offset, -77)
    MinY.location = (-732-offset, -172)
    MaxX.location = (-732-offset, -295)
    MaxY.location = (-732-offset, -378)