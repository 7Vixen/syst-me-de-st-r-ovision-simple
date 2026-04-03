import bpy
import bmesh
import statistics

file_path = "C:/Users/rog wephyrus/Desktop/Vison/ProjetVison/nuage_points.xyz"

# Nettoyage
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# Lecture
points_raw = []
with open(file_path, 'r') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 3:
            try:
                points_raw.append((float(parts[0]), float(parts[1]), float(parts[2])))
            except:
                continue

print(f"[INFO] Points bruts : {len(points_raw)}")

# Filtrage outliers (IQR sur Z)
zs = sorted([p[2] for p in points_raw])
q1 = zs[len(zs)//4]
q3 = zs[3*len(zs)//4]
iqr = q3 - q1
z_min = q1 - 1.5 * iqr
z_max = q3 + 1.5 * iqr
points_filtered = [p for p in points_raw if z_min <= p[2] <= z_max]
print(f"[INFO] Après filtrage : {len(points_filtered)} points")

# Normalisation
all_vals = [v for p in points_filtered for v in p]
cx = sum(p[0] for p in points_filtered) / len(points_filtered)
cy = sum(p[1] for p in points_filtered) / len(points_filtered)
cz = sum(p[2] for p in points_filtered) / len(points_filtered)
spread = max(all_vals) - min(all_vals)
SCALE = 5.0 / spread

# Conversion OpenCV → Blender (Z=profondeur → Y, Y→Z)
points = [
    ((p[0]-cx)*SCALE, -(p[2]-cz)*SCALE, -(p[1]-cy)*SCALE)
    for p in points_filtered
]

# Création du mesh
mesh = bpy.data.meshes.new("NuagePoints")
obj  = bpy.data.objects.new("NuagePoints", mesh)
bpy.context.collection.objects.link(obj)
bpy.context.view_layer.objects.active = obj
obj.select_set(True)

bm = bmesh.new()
for p in points:
    bm.verts.new(p)
bm.verts.ensure_lookup_table()
bm.to_mesh(mesh)
bm.free()
mesh.update()

# Geometry Nodes pour afficher les sphères
gn_mod = obj.modifiers.new("PointCloud", 'NODES')
node_group = bpy.data.node_groups.new("PointCloudNodes", 'GeometryNodeTree')
gn_mod.node_group = node_group

# Déclaration des interfaces entrée/sortie (Blender 4.x)
node_group.interface.new_socket("Geometry", in_out='INPUT',  socket_type='NodeSocketGeometry')
node_group.interface.new_socket("Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')

nt = gn_mod.node_group
nt.nodes.clear()

input_node  = nt.nodes.new('NodeGroupInput')
output_node = nt.nodes.new('NodeGroupOutput')
inst_node   = nt.nodes.new('GeometryNodeInstanceOnPoints')
sphere_node = nt.nodes.new('GeometryNodeMeshUVSphere')
realize     = nt.nodes.new('GeometryNodeRealizeInstances')

sphere_node.inputs['Rings'].default_value    = 4
sphere_node.inputs['Segments'].default_value = 6
sphere_node.inputs['Radius'].default_value   = 0.015

input_node.location  = (-400, 0)
sphere_node.location = (-200, -150)
inst_node.location   = (0, 0)
realize.location     = (200, 0)
output_node.location = (400, 0)

links = nt.links
links.new(input_node.outputs['Geometry'],  inst_node.inputs['Points'])
links.new(sphere_node.outputs['Mesh'],     inst_node.inputs['Instance'])
links.new(inst_node.outputs['Instances'], realize.inputs['Geometry'])
links.new(realize.outputs['Geometry'],    output_node.inputs['Geometry'])

# Matériau bleu
mat = bpy.data.materials.new("MatPoints")
mat.use_nodes = True
bsdf = mat.node_tree.nodes.get("Principled BSDF")
if bsdf:
    bsdf.inputs["Base Color"].default_value = (0.1, 0.6, 1.0, 1.0)
    bsdf.inputs["Roughness"].default_value  = 0.3
obj.data.materials.append(mat)

bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')

print("[DONE] Appuie sur Numpad . pour centrer la vue !")