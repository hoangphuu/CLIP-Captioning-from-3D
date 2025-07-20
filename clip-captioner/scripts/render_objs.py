import os
import trimesh
import pyrender
import numpy as np
from PIL import Image
from tqdm import tqdm


def render_obj_to_image(obj_path, save_path, resolution=(224, 224)):
    mesh = trimesh.load(obj_path, force='mesh')

    if not isinstance(mesh, trimesh.Trimesh):
        return False

    scene = pyrender.Scene()
    mesh = pyrender.Mesh.from_trimesh(mesh)
    scene.add(mesh)

    # Add camera + light
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    cam_pose = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, -0.1],
        [0.0, 0.0, 1.0, 0.3],
        [0.0, 0.0, 0.0, 1.0],
    ])
    scene.add(camera, pose=cam_pose)
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
    scene.add(light, pose=cam_pose)

    r = pyrender.OffscreenRenderer(*resolution)
    color, _ = r.render(scene)
    r.delete()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    Image.fromarray(color).save(save_path)
    return True


def render_all_objs(
    obj_folder="./data/pix3d/model",
    output_folder="./data/pix3d/renders"
):
    obj_files = [f for f in os.listdir(obj_folder) if f.endswith(".obj")]

    for obj_file in tqdm(obj_files, desc="Rendering .obj files"):
        obj_path = os.path.join(obj_folder, obj_file)
        output_path = os.path.join(output_folder, obj_file.replace(".obj", ".png"))
        if not os.path.exists(output_path):
            render_obj_to_image(obj_path, output_path)

    print(f"✅ Rendered all .obj files to {output_folder}")

if __name__ == "__main__":
    render_all_objs()
