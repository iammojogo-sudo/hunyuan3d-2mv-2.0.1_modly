"""PBR texture applicator bridge.

Takes an already-UV-unwrapped mesh (GLB) plus texture atlas image(s) and
exports a PBR GLB: albedo + normal + metallic/roughness. Pure trimesh +
PIL + pygltflib + numpy: no GPU.

Inputs (JSON arg):
  mesh_path:              GLB that already has UV coordinates
  texture_path:           albedo atlas (PNG/JPG). Optional if texture_folder
                          holds a detectable albedo file.
  texture_folder:         folder scanned for maps by filename keyword
  normal_path:            tangent-space normal map (overrides folder scan)
  rough_path:             roughness map (0=mirror smooth, 1=matte)
  metallic_path:          metallic map (0=dielectric, 1=metal)
  normal_scale:           glTF normalTexture scale
  normal_flip_y:          flip green channel (DirectX-style normal maps)
  smooth_normals:         recompute vertex normals for smooth shading
  output_path:            where to write the PBR GLB
"""
import json
import os
import sys

import numpy as np
import trimesh
from PIL import Image


def _report(pct, step, subtext=""):
    print(json.dumps({
        "type": "progress", "pct": pct, "step": step, "subtext": subtext
    }), flush=True)


def _log(msg):
    print(json.dumps({"type": "log", "message": msg}), flush=True)


def _flatten_scene(scene):
    geoms = [g for g in scene.geometry.values() if hasattr(g, "faces") and len(getattr(g, "faces", []))]
    if not geoms:
        return None
    if len(geoms) == 1:
        return geoms[0]
    try:
        return trimesh.util.concatenate(geoms)
    except Exception:
        return geoms[0]


_ALBEDO_KEYS = ("texturemap", "albedo", "basecolor", "base_color", "base-color", "diffuse")
_NORMAL_KEYS = ("normalmap", "normal", "nrm", "nor_")
_METAL_KEYS = ("metallicmap", "metallic", "metal", "specular", "spec")
_ROUGH_KEYS = ("roughnessmap", "roughness", "rough")
_IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")


def _scan_folder(folder):
    """Detect maps in a folder by filename keyword. Returns {kind: path}."""
    found = {}
    try:
        names = sorted(os.listdir(folder))
    except OSError:
        return found
    imgs = [n for n in names if n.lower().endswith(_IMG_EXTS)]
    for n in imgs:
        low = n.lower()
        p = os.path.join(folder, n)
        if any(k in low for k in _NORMAL_KEYS):
            found.setdefault("normal", p)
        elif any(k in low for k in _METAL_KEYS):
            found.setdefault("metallic", p)
        elif any(k in low for k in _ROUGH_KEYS):
            found.setdefault("rough", p)
        elif any(k in low for k in _ALBEDO_KEYS):
            found.setdefault("albedo", p)
    if "albedo" not in found and len(imgs) == 1:
        found["albedo"] = os.path.join(folder, imgs[0])
    return found


def _attach_pbr(temp_glb, output_path, normal_arr=None, rough_arr=None,
                normal_scale=1.0, metal_arr=None):
    """Attach normal + ORM textures to every material in a GLB."""
    from pygltflib import GLTF2, BufferView, Image as GImage, Texture as GTexture
    from pygltflib import NormalMaterialTexture, TextureInfo
    import io as _io

    g = GLTF2().load(temp_glb)

    def _png_bytes(arr):
        buf = _io.BytesIO()
        Image.fromarray(arr).save(buf, format="PNG")
        return buf.getvalue()

    def _embed_image(data, name):
        blob = bytearray(g.binary_blob() or b"")
        blob += b"\0" * (-len(blob) % 4)
        off = len(blob)
        blob += data
        g.set_binary_blob(bytes(blob))
        g.buffers[0].byteLength = len(blob)
        bv = BufferView(buffer=0, byteOffset=off, byteLength=len(data), name=name)
        g.bufferViews.append(bv)
        img = GImage()
        img.bufferView = len(g.bufferViews) - 1
        img.mimeType = "image/png"
        img.name = name
        g.images.append(img)
        return len(g.images) - 1

    if normal_arr is not None:
        _embed_image(_png_bytes(normal_arr), "normal")
        nt = GTexture()
        nt.source = len(g.images) - 1
        nt.name = "normal"
        g.textures.append(nt)
        normal_idx = len(g.textures) - 1
    else:
        normal_idx = None

    if rough_arr is not None:
        H, W = rough_arr.shape
        metal = metal_arr if metal_arr is not None else np.zeros((H, W), dtype=np.float64)
        orm = np.zeros((H, W, 3), dtype=np.uint8)
        orm[:, :, 0] = 255  # occlusion: none (full white R)
        orm[:, :, 1] = (np.clip(rough_arr, 0, 1) * 255).astype(np.uint8)
        orm[:, :, 2] = (np.clip(metal, 0, 1) * 255).astype(np.uint8)
        _embed_image(_png_bytes(orm), "metallicRoughness")
        mt = GTexture()
        mt.source = len(g.images) - 1
        mt.name = "metallicRoughness"
        g.textures.append(mt)
        orm_idx = len(g.textures) - 1
    else:
        orm_idx = None

    for mat in g.materials:
        if normal_idx is not None:
            nm = NormalMaterialTexture()
            nm.index = normal_idx
            nm.scale = float(normal_scale)
            mat.normalTexture = nm
        if orm_idx is not None:
            ti = TextureInfo()
            ti.index = orm_idx
            mat.pbrMetallicRoughness.metallicRoughnessTexture = ti
            mat.pbrMetallicRoughness.metallicFactor = 1.0
            mat.pbrMetallicRoughness.roughnessFactor = 1.0

    g.save(output_path)


def apply_texture(args):
    mesh_path = args.get("mesh_path", "")
    texture_path = args.get("texture_path", "")
    output_path = args.get("output_path", "output.glb")

    if not mesh_path or not os.path.exists(mesh_path):
        raise RuntimeError(f"mesh_path not found: {mesh_path}")

    mode = str(args.get("mode", "pbr") or "pbr").lower()
    albedo_only = (mode == "albedo")
    smooth_normals = str(args.get("smooth_normals", "on") or "on").lower() == "on"

    # Resolve maps: explicit paths win, otherwise scan the folder.
    folder = "" if albedo_only else (args.get("texture_folder", "") or "")
    found = _scan_folder(folder) if folder and os.path.isdir(folder) else {}
    if folder and not os.path.isdir(folder):
        _log(f"[texture_apply] texture folder not found: {folder}")
    if (not texture_path or not os.path.exists(texture_path)) and found.get("albedo"):
        texture_path = found["albedo"]
    if not texture_path or not os.path.exists(texture_path):
        raise RuntimeError(
            "texture_apply: no albedo texture provided — wire an image or "
            "set image_path")

    def _pick(key, *names):
        for n in names:
            v = args.get(n, "")
            if v and os.path.exists(v):
                return v
        return found.get(key, "")

    normal_path = "" if albedo_only else _pick("normal", "normal_path")
    rough_path = "" if albedo_only else _pick("rough", "rough_path", "specular_path")
    metal_path = "" if albedo_only else _pick("metallic", "metallic_path", "metal_path")

    normal_scale = float(args.get("normal_scale", 1.0) or 0)
    if normal_scale <= 0:
        normal_scale = 1.0
    normal_flip_y = str(args.get("normal_flip_y", "off") or "off").lower() == "on"

    _log(f"[texture_apply] mode={mode} albedo={os.path.basename(texture_path)}"
         + (f" normal={os.path.basename(normal_path)}" if normal_path else "")
         + (f" rough={os.path.basename(rough_path)}" if rough_path else "")
         + (f" metal={os.path.basename(metal_path)}" if metal_path else ""))

    _report(20, "Loading mesh", os.path.basename(mesh_path))
    loaded = trimesh.load(mesh_path, force=None)
    mesh = _flatten_scene(loaded) if isinstance(loaded, trimesh.Scene) else loaded
    if mesh is None or not hasattr(mesh, "faces") or len(mesh.faces) == 0:
        raise RuntimeError("Mesh has no usable geometry")
    _log(f"[texture_apply] mesh: {len(mesh.faces)} faces, {len(mesh.vertices)} verts")

    if isinstance(mesh, trimesh.Scene):
        mesh = _flatten_scene(mesh)
        if mesh is None:
            raise RuntimeError("Mesh has no usable geometry")

    uv = getattr(mesh.visual, "uv", None)
    if uv is None or len(uv) == 0:
        raise RuntimeError(
            f"Mesh '{os.path.basename(mesh_path)}' has no UV coordinates — it "
            "must be UV-unwrapped before textures can be applied. Run the "
            "'Texture Mesh' node first.")
    uv = np.asarray(uv, dtype=np.float64)
    _log(f"[texture_apply] UVs OK: {len(uv)} coords")

    _report(50, "Loading albedo", os.path.basename(texture_path))
    tex = Image.open(texture_path).convert("RGB")
    _log(f"[texture_apply] albedo: {tex.size[0]}x{tex.size[1]}")

    # Normal map (optional).
    normal_arr = None
    if normal_path and os.path.exists(normal_path):
        _report(60, "Loading normal map", os.path.basename(normal_path))
        normal_arr = np.asarray(Image.open(normal_path).convert("RGB"))
        if normal_flip_y:
            normal_arr = normal_arr.copy()
            normal_arr[:, :, 1] = 255 - normal_arr[:, :, 1]
        _log(f"[texture_apply] normal: {normal_arr.shape[1]}x{normal_arr.shape[0]}"
             + (" (Y flipped)" if normal_flip_y else ""))
    elif folder:
        _log("[texture_apply] no normal map detected")

    # Roughness (optional).
    rough_arr = None
    if rough_path and os.path.exists(rough_path):
        _report(65, "Loading roughness", os.path.basename(rough_path))
        gray = np.asarray(Image.open(rough_path).convert("L"), dtype=np.float64) / 255.0
        rough_arr = gray
        _log(f"[texture_apply] roughness: {gray.shape[1]}x{gray.shape[0]}")
    elif folder:
        _log("[texture_apply] no roughness map detected")

    # Metallic (optional).
    metal_arr = None
    if metal_path and os.path.exists(metal_path):
        _report(66, "Loading metallic", os.path.basename(metal_path))
        metal_arr = np.asarray(Image.open(metal_path).convert("L"), dtype=np.float64) / 255.0
        _log(f"[texture_apply] metallic: {metal_arr.shape[1]}x{metal_arr.shape[0]}")
    elif folder:
        _log("[texture_apply] no metallic map detected")

    _report(70, "Applying textures to mesh", "mapping images onto UVs")

    material = trimesh.visual.texture.SimpleMaterial(
        image=tex, diffuse=(255, 255, 255))
    mesh.visual = trimesh.visual.texture.TextureVisuals(
        uv=uv, image=tex, material=material)

    _report(85, "Exporting PBR mesh", "writing GLB")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # When PBR mode is on but no maps were provided, generate a default glossy
    # ORM so the viewer sees PBR material properties (not flat albedo-only).
    if not albedo_only and rough_arr is None and normal_arr is None and metal_arr is None:
        _log("[texture_apply] PBR mode with no maps — applying default glossy (rough=0.3)")
        H, W = tex.size[1], tex.size[0]
        rough_arr = np.full((H, W), 0.3, dtype=np.float64)
        metal_arr = np.zeros((H, W), dtype=np.float64)

    # Determine output filename based on what maps were found.
    _has_pbr_maps = normal_arr is not None or rough_arr is not None or metal_arr is not None
    if _has_pbr_maps:
        if os.path.basename(output_path) == "partial_pbr_mesh.glb":
            output_path = output_path.replace("partial_pbr_mesh.glb", "pbr_mesh.glb")
    else:
        if os.path.basename(output_path) == "pbr_mesh.glb":
            output_path = output_path.replace("pbr_mesh.glb", "partial_pbr_mesh.glb")

    # Export with PBR maps via pygltflib.
    if normal_arr is not None or rough_arr is not None:
        import tempfile as _tf
        _fd, _tmp = _tf.mkstemp(suffix=".glb", prefix="hunyuan_pbr_base_")
        os.close(_fd)
        try:
            mesh.export(_tmp)
            _attach_pbr(_tmp, str(output_path),
                        normal_arr=normal_arr, rough_arr=rough_arr,
                        normal_scale=normal_scale,
                        metal_arr=metal_arr)
            attached = ([ "normal" ] if normal_arr is not None else []) + \
                       ([ "metallicRoughness" ] if rough_arr is not None else [])
            _log(f"[texture_apply] PBR slots: albedo + {', '.join(attached)}")
        finally:
            try:
                os.remove(_tmp)
            except OSError:
                pass
    else:
        mesh.export(output_path)

    # Smooth normals: apply AFTER export so pygltflib's re-save doesn't strip
    # them. Load the final GLB, compute smooth normals, inject into the GLB.
    if smooth_normals:
        try:
            from pygltflib import GLTF2 as _GLTF2
            _g = _GLTF2().load(str(output_path))
            _verts = np.asarray(mesh.vertices, dtype=np.float64)
            _faces = np.asarray(mesh.faces)
            _fn = trimesh.triangles.normals(_verts[_faces])[0]
            _vn = trimesh.geometry.mean_vertex_normals(
                vertex_count=_verts.shape[0], faces=_faces, face_normals=_fn)
            import io as _io
            _norm_bytes = _vn.astype(np.float32).tobytes()
            _blob = bytearray(_g.binary_blob() or b"")
            _blob += b"\0" * (-len(_blob) % 4)
            _off = len(_blob)
            _blob += _norm_bytes
            _g.set_binary_blob(bytes(_blob))
            _g.buffers[0].byteLength = len(_blob)
            from pygltflib import BufferView as _BV, Accessor as _Acc
            _bv = _BV(buffer=0, byteOffset=_off, byteLength=len(_norm_bytes), target=34962)
            _g.bufferViews.append(_bv)
            _acc = _Acc(bufferView=len(_g.bufferViews) - 1,
                        componentType=5126, count=_verts.shape[0], type="VEC3")
            _g.accessors.append(_acc)
            for _m in _g.meshes:
                for _p in _m.primitives:
                    _p.attributes.NORMAL = len(_g.accessors) - 1
            _g.save(str(output_path))
            _log(f"[texture_apply] smooth normals applied ({_verts.shape[0]} vertices)")
        except Exception as _sn_e:
            _log(f"[texture_apply] smooth normals skipped: {_sn_e}")
    else:
        _log("[texture_apply] smooth normals off — keeping flat/faceted normals")

    _report(100, "Done", "")
    print(json.dumps({"type": "done", "output_path": output_path}), flush=True)


if __name__ == "__main__":
    raw = sys.argv[1] if len(sys.argv) > 1 else "{}"
    if os.path.isfile(raw):
        with open(raw, "r", encoding="utf-8") as f:
            args = json.load(f)
    else:
        args = json.loads(raw)
    try:
        apply_texture(args)
    except Exception as e:
        print(json.dumps({"type": "error", "message": str(e)}), flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)
