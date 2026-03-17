"""Generate a simple Blender scene with a field, a tree, and a drone.

Run with Blender:

    blender -b -P scripts/generate_simple_blender_scene.py -- \
        --output artifacts/simple_field_tree_drone.blend
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import bpy
from mathutils import Vector


def parse_args() -> argparse.Namespace:
    argv = sys.argv
    args = argv[argv.index("--") + 1 :] if "--" in argv else []

    parser = argparse.ArgumentParser(
        description="Create a simple field/tree/drone Blender scene."
    )
    parser.add_argument(
        "--output",
        default="artifacts/simple_field_tree_drone.blend",
        help="Path to write the generated .blend file.",
    )
    return parser.parse_args(args)


def make_principled_material(
    name: str,
    base_color: tuple[float, float, float, float],
    *,
    roughness: float = 0.5,
    metallic: float = 0.0,
) -> bpy.types.Material:
    material = bpy.data.materials.new(name=name)
    material.use_nodes = True
    principled = material.node_tree.nodes["Principled BSDF"]
    principled.inputs["Base Color"].default_value = base_color
    principled.inputs["Roughness"].default_value = roughness
    principled.inputs["Metallic"].default_value = metallic
    return material


def assign_material(obj: bpy.types.Object, material: bpy.types.Material) -> None:
    if obj.data.materials:
        obj.data.materials[0] = material
    else:
        obj.data.materials.append(material)


def set_object_origin_to_geometry(obj: bpy.types.Object) -> None:
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
    obj.select_set(False)


def create_world() -> None:
    world = bpy.data.worlds.new(name="World")
    bpy.context.scene.world = world
    world.use_nodes = True
    background = world.node_tree.nodes["Background"]
    background.inputs["Color"].default_value = (0.52, 0.74, 0.96, 1.0)
    background.inputs["Strength"].default_value = 0.9


def create_field(material: bpy.types.Material) -> bpy.types.Object:
    bpy.ops.mesh.primitive_plane_add(size=60, location=(0.0, 0.0, 0.0))
    field = bpy.context.object
    field.name = "Field"
    assign_material(field, material)
    bpy.ops.object.shade_smooth()
    return field


def create_tree(
    trunk_material: bpy.types.Material,
    canopy_material: bpy.types.Material,
) -> bpy.types.Object:
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=24,
        radius=0.25,
        depth=2.8,
        location=(-4.0, 1.0, 1.4),
    )
    trunk = bpy.context.object
    trunk.name = "TreeTrunk"
    assign_material(trunk, trunk_material)
    bpy.ops.object.shade_smooth()

    canopy_offsets = [
        Vector((-4.0, 1.0, 3.2)),
        Vector((-4.5, 0.6, 3.0)),
        Vector((-3.5, 1.3, 2.9)),
        Vector((-4.1, 1.4, 3.5)),
    ]
    canopy_radii = [1.3, 1.0, 0.9, 0.85]

    canopy_parts: list[bpy.types.Object] = []
    for index, (offset, radius) in enumerate(zip(canopy_offsets, canopy_radii)):
        bpy.ops.mesh.primitive_ico_sphere_add(
            subdivisions=3,
            radius=radius,
            location=offset,
        )
        canopy = bpy.context.object
        canopy.name = f"TreeCanopy{index + 1}"
        assign_material(canopy, canopy_material)
        bpy.ops.object.shade_smooth()
        canopy_parts.append(canopy)

    bpy.ops.object.select_all(action="DESELECT")
    trunk.select_set(True)
    for canopy in canopy_parts:
        canopy.select_set(True)
    bpy.context.view_layer.objects.active = trunk
    bpy.ops.object.join()
    tree = bpy.context.object
    tree.name = "Tree"
    return tree


def add_arm(
    start: Vector,
    end: Vector,
    material: bpy.types.Material,
) -> bpy.types.Object:
    direction = end - start
    midpoint = start + direction / 2
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=16,
        radius=0.05,
        depth=direction.length,
        location=midpoint,
    )
    arm = bpy.context.object
    arm.rotation_mode = "QUATERNION"
    arm.rotation_quaternion = Vector((0.0, 0.0, 1.0)).rotation_difference(
        direction.normalized()
    )
    assign_material(arm, material)
    bpy.ops.object.shade_smooth()
    return arm


def create_drone(
    body_material: bpy.types.Material,
    metal_material: bpy.types.Material,
    rotor_material: bpy.types.Material,
) -> bpy.types.Object:
    drone_origin = Vector((3.5, -2.5, 2.3))

    bpy.ops.mesh.primitive_cube_add(size=1.0, location=drone_origin)
    body = bpy.context.object
    body.name = "DroneBody"
    body.scale = (0.42, 0.30, 0.12)
    assign_material(body, body_material)
    bpy.ops.object.shade_smooth()

    arm_targets = [
        drone_origin + Vector((0.9, 0.9, 0.0)),
        drone_origin + Vector((0.9, -0.9, 0.0)),
        drone_origin + Vector((-0.9, 0.9, 0.0)),
        drone_origin + Vector((-0.9, -0.9, 0.0)),
    ]

    parts: list[bpy.types.Object] = [body]
    for index, target in enumerate(arm_targets):
        arm = add_arm(drone_origin, target, metal_material)
        arm.name = f"DroneArm{index + 1}"
        parts.append(arm)

        bpy.ops.mesh.primitive_cylinder_add(
            vertices=24,
            radius=0.18,
            depth=0.03,
            location=target + Vector((0.0, 0.0, 0.04)),
        )
        rotor = bpy.context.object
        rotor.name = f"DroneRotor{index + 1}"
        assign_material(rotor, rotor_material)
        parts.append(rotor)

        bpy.ops.mesh.primitive_cylinder_add(
            vertices=18,
            radius=0.08,
            depth=0.08,
            location=target,
        )
        motor = bpy.context.object
        motor.name = f"DroneMotor{index + 1}"
        assign_material(motor, metal_material)
        bpy.ops.object.shade_smooth()
        parts.append(motor)

    bpy.ops.mesh.primitive_uv_sphere_add(
        segments=24,
        ring_count=12,
        radius=0.11,
        location=drone_origin + Vector((0.44, 0.0, -0.08)),
    )
    camera_pod = bpy.context.object
    camera_pod.name = "DroneCameraPod"
    assign_material(camera_pod, metal_material)
    bpy.ops.object.shade_smooth()
    parts.append(camera_pod)

    bpy.ops.object.select_all(action="DESELECT")
    body.select_set(True)
    for part in parts[1:]:
        part.select_set(True)
    bpy.context.view_layer.objects.active = body
    bpy.ops.object.join()
    drone = bpy.context.object
    drone.name = "Drone"
    set_object_origin_to_geometry(drone)
    return drone


def create_light() -> bpy.types.Object:
    bpy.ops.object.light_add(type="SUN", location=(0.0, 0.0, 8.0))
    sun = bpy.context.object
    sun.name = "Sun"
    sun.data.energy = 3.0
    sun.rotation_euler = (math.radians(50), 0.0, math.radians(35))
    return sun


def create_camera(target: Vector) -> bpy.types.Object:
    bpy.ops.object.camera_add(location=(11.0, -13.0, 6.8))
    camera = bpy.context.object
    camera.name = "Camera"

    direction = target - camera.location
    camera.rotation_mode = "QUATERNION"
    camera.rotation_quaternion = direction.to_track_quat("-Z", "Y")
    camera.data.lens = 35
    return camera


def configure_scene(camera: bpy.types.Object) -> None:
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 128
    scene.render.resolution_x = 1920
    scene.render.resolution_y = 1080
    scene.render.film_transparent = False
    scene.camera = camera


def main() -> None:
    args = parse_args()
    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    create_world()

    grass_material = make_principled_material(
        "GrassMaterial",
        (0.16, 0.44, 0.14, 1.0),
        roughness=0.95,
    )
    trunk_material = make_principled_material(
        "TrunkMaterial",
        (0.27, 0.16, 0.08, 1.0),
        roughness=0.9,
    )
    leaf_material = make_principled_material(
        "LeafMaterial",
        (0.11, 0.38, 0.10, 1.0),
        roughness=0.8,
    )
    drone_body_material = make_principled_material(
        "DroneBodyMaterial",
        (0.18, 0.19, 0.22, 1.0),
        roughness=0.35,
        metallic=0.15,
    )
    metal_material = make_principled_material(
        "MetalMaterial",
        (0.45, 0.47, 0.50, 1.0),
        roughness=0.28,
        metallic=0.85,
    )
    rotor_material = make_principled_material(
        "RotorMaterial",
        (0.06, 0.06, 0.06, 1.0),
        roughness=0.55,
    )

    create_field(grass_material)
    create_tree(trunk_material, leaf_material)
    create_drone(drone_body_material, metal_material, rotor_material)
    create_light()
    camera = create_camera(Vector((0.0, 0.0, 1.8)))
    configure_scene(camera)

    bpy.ops.wm.save_as_mainfile(filepath=output_path)
    print(f"Saved Blender scene to {output_path}")


if __name__ == "__main__":
    main()
