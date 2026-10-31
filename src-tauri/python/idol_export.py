"""Export Idol's 3D Gaussian splats (.ply) to other formats.

Gaussian splats:  ply (standard 3DGS), spz (Niantic, compressed, converted to its y-up frame), splat
                  (antimatter15, in Idol's PLY frame)
Point cloud:      points (.ply with xyz + rgb, y-up)
Mesh:             glb, obj, stl, mesh-ply. The default uses screened Poisson reconstruction (Open3D) from
                  the Gaussian centres and their smallest-axis normals, then keeps the largest surface,
                  smooths, decimates, and colours vertices from the nearest splats. This preserves faces,
                  eyes and thin parts much better than a density field. `--method density` falls back to
                  an opacity density grid + marching cubes for very thin or unusual objects.

Game-ready mesh:  game-glb. A low-poly version of the Poisson mesh (default 20k triangles) with automatic UVs
                  (xatlas) and the splat colours baked into a texture, with smooth normals and a matte PBR
                  material: imports straight into Blender, Unity, Unreal and Godot, ready for rigging.

Mesh formats are turned upright: y-up for glb / obj / mesh-ply, z-up for stl (what slicers expect).

Standalone use: python idol_export.py in.ply out.glb [--resolution 256] [--faces 200000] ...
"""
import gzip
import os
import shutil
import struct
import sys
import time

import numpy as np

SH_C0 = 0.28209479177387814
# Colours are sampled this far (x the object size) outside the surface. Solid parts are filled with darker
# interior splats that the viewer never shows; sampling right on the surface mixes them in.
COLOUR_OFFSET = 0.004
SPLAT_FORMATS = ('ply', 'spz', 'splat')
MESH_FORMATS = ('glb', 'obj', 'stl', 'mesh-ply', 'game-glb')
FORMATS = SPLAT_FORMATS + ('points',) + MESH_FORMATS
EXTENSIONS = {'ply': '.ply', 'spz': '.spz', 'splat': '.splat', 'points': '_points.ply', 'glb': '.glb',
              'obj': '.obj', 'stl': '.stl', 'mesh-ply': '_mesh.ply', 'game-glb': '_game.glb'}


def default_path(ply_path, fmt):
    base = ply_path[:-4] if ply_path.lower().endswith('.ply') else ply_path
    return base + ('_splat.ply' if fmt == 'ply' else EXTENSIONS[fmt])


# ---------------------------------------------------------------------------------------------------------------
# Reading

class Splats:
    def __init__(self, xyz, f_dc, opacity_logit, log_scale, rot):
        self.xyz = xyz.astype(np.float32)                     # (N, 3)
        self.f_dc = f_dc.astype(np.float32)                   # (N, 3) SH degree-0 coefficients
        self.opacity_logit = opacity_logit.astype(np.float32)  # (N,)
        self.log_scale = log_scale.astype(np.float32)         # (N, 3)
        rot = rot.astype(np.float32)                          # (N, 4) w, x, y, z
        self.rot = rot / np.maximum(np.linalg.norm(rot, axis=1, keepdims=True), 1e-12)

    @property
    def count(self):
        return len(self.xyz)

    @property
    def rgb(self):
        return np.clip(0.5 + SH_C0 * self.f_dc, 0.0, 1.0)

    @property
    def alpha(self):
        return 1.0 / (1.0 + np.exp(-self.opacity_logit))

    @property
    def scale(self):
        return np.exp(self.log_scale)


def read_ply(path):
    from plyfile import PlyData
    vertex = PlyData.read(path)['vertex']
    names = vertex.data.dtype.names
    for required in ('x', 'f_dc_0', 'opacity', 'scale_0', 'rot_0'):
        if required not in names:
            raise ValueError(f'{path} is not a 3D Gaussian splat PLY (no {required})')

    def cols(*keys):
        return np.stack([np.asarray(vertex[k], dtype=np.float32) for k in keys], axis=1)

    return Splats(cols('x', 'y', 'z'), cols('f_dc_0', 'f_dc_1', 'f_dc_2'), np.asarray(vertex['opacity'], np.float32),
                  cols('scale_0', 'scale_1', 'scale_2'), cols('rot_0', 'rot_1', 'rot_2', 'rot_3'))


# ---------------------------------------------------------------------------------------------------------------
# Splat formats

def write_splat(splats, path):
    """antimatter15 .splat: 32 bytes per splat, most important first."""
    scale = splats.scale
    importance = splats.alpha * scale.prod(axis=1)
    order = np.argsort(-importance)
    rgba = np.concatenate([splats.rgb, splats.alpha[:, None]], axis=1)
    out = np.zeros(splats.count, dtype=[('xyz', '<f4', 3), ('scale', '<f4', 3), ('rgba', 'u1', 4), ('rot', 'u1', 4)])
    out['xyz'] = splats.xyz[order]
    out['scale'] = scale[order]
    out['rgba'] = np.clip(rgba[order] * 255.0 + 0.5, 0, 255).astype(np.uint8)
    out['rot'] = np.clip(splats.rot[order] * 128.0 + 128.0, 0, 255).astype(np.uint8)
    with open(path, 'wb') as f:
        f.write(out.tobytes())


def write_spz(splats, path, fractional_bits=12):
    """Niantic .spz version 2 (gzip; 24-bit fixed-point positions, 8-bit everything else, no SH beyond DC).

    SPZ stores gaussians in its RUB frame (y up, z back), while Idol's PLY is RDF (y down, z front).
    Convert both the centres and the rotations, or the splat loads upside down in SPZ viewers."""
    n = splats.count
    header = struct.pack('<IIIBBBB', 0x5053474E, 2, n, 0, fractional_bits, 0, 0)
    xyz = np.stack([splats.xyz[:, 0], -splats.xyz[:, 1], -splats.xyz[:, 2]], axis=1)
    fixed = np.round(xyz.astype(np.float64) * (1 << fractional_bits)).astype(np.int64)
    fixed = np.clip(fixed, -(1 << 23), (1 << 23) - 1) & 0xFFFFFF
    positions = np.stack([fixed & 0xFF, (fixed >> 8) & 0xFF, (fixed >> 16) & 0xFF], axis=-1).astype(np.uint8)
    alphas = np.clip(splats.alpha * 255.0 + 0.5, 0, 255).astype(np.uint8)
    colors = np.clip(splats.f_dc * (0.15 * 255.0) + 127.5 + 0.5, 0, 255).astype(np.uint8)
    scales = np.clip((splats.log_scale + 10.0) * 16.0 + 0.5, 0, 255).astype(np.uint8)
    rot = splats.rot.copy()
    rot[:, 2] *= -1  # RDF -> RUB flips quaternion y and z
    rot[:, 3] *= -1
    rot[rot[:, 0] < 0] *= -1  # w >= 0, so w can be rebuilt from x, y, z
    rotations = np.clip(rot[:, 1:] * 127.5 + 127.5 + 0.5, 0, 255).astype(np.uint8)
    body = b''.join([header, positions.tobytes(), alphas.tobytes(), colors.tobytes(), scales.tobytes(),
                     rotations.tobytes()])
    with open(path, 'wb') as f:
        f.write(gzip.compress(body, compresslevel=9))


def write_points(splats, path, min_alpha=0.1):
    from plyfile import PlyData, PlyElement
    keep = splats.alpha >= min_alpha
    xyz, rgb = splats.xyz[keep], (splats.rgb[keep] * 255 + 0.5).astype(np.uint8)
    xyz = upright(xyz)
    data = np.zeros(len(xyz), dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'),
                                     ('blue', 'u1')])
    data['x'], data['y'], data['z'] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    data['red'], data['green'], data['blue'] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    PlyData([PlyElement.describe(data, 'vertex')]).write(path)


def upright(xyz):
    """Idol's PLY is in the y-down frame 3DGS viewers use; meshes and point clouds are written y-up."""
    return np.stack([xyz[:, 0], -xyz[:, 1], -xyz[:, 2]], axis=1)


# ---------------------------------------------------------------------------------------------------------------
# Mesh

def quaternion_to_matrix(q):
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], axis=1).reshape(-1, 3, 3)


def density_grids(splats, resolution, progress=None):
    """Density D(x) = sum_i alpha_i * exp(-0.5 d^T S_i^-1 d) and colour C(x) = sum_i alpha_i g_i rgb_i / D on a
    cubic grid around the object. Runs on the GPU when there is one."""
    import torch

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    keep = splats.alpha >= 0.02
    xyz, alpha, scale = splats.xyz[keep], splats.alpha[keep], np.maximum(splats.scale[keep], 1e-5)
    rgb, rot = splats.rgb[keep], splats.rot[keep]

    # Grid bounds: robust extent of the confident splats plus a margin.
    confident = xyz[alpha >= 0.3] if (alpha >= 0.3).sum() > 100 else xyz
    lo, hi = np.percentile(confident, 0.2, axis=0), np.percentile(confident, 99.8, axis=0)
    margin = (hi - lo).max() * 0.06
    lo, hi = lo - margin, hi + margin
    size = (hi - lo).max()
    center = (lo + hi) / 2
    lo = center - size / 2
    voxel = size / (resolution - 1)

    # Inverse covariances: R diag(1/s^2) R^T.
    R = quaternion_to_matrix(rot)
    inv_cov = np.einsum('nij,nj,nkj->nik', R, 1.0 / scale ** 2, R).astype(np.float32)
    # Each Gaussian touches voxels within 3 sigma of its longest axis (capped for speed).
    radius = np.clip(np.ceil(3.0 * scale.max(axis=1) / voxel), 1, 10).astype(np.int64)

    t = lambda a: torch.as_tensor(a, device=device)
    xyz_t, alpha_t, rgb_t, inv_t = t(xyz), t(alpha), t(rgb), t(inv_cov)
    lo_t = t(lo.astype(np.float32))
    density = torch.zeros(resolution ** 3, device=device)
    colour = torch.zeros(resolution ** 3, 3, device=device)

    # Process Gaussians grouped by footprint radius so each batch uses one offset cube.
    radii = np.unique(radius)
    done, total = 0, len(xyz)
    for r in radii:
        idx_all = np.nonzero(radius == r)[0]
        span = torch.arange(-r, r + 1, device=device)
        offsets = torch.stack(torch.meshgrid(span, span, span, indexing='ij'), dim=-1).reshape(-1, 3)  # (K, 3)
        batch = max(1, int(6_000_000 // len(offsets)))
        for start in range(0, len(idx_all), batch):
            idx = t(idx_all[start:start + batch])
            mu = xyz_t[idx]                                                       # (B, 3)
            base = torch.round((mu - lo_t) / voxel).long()                        # nearest voxel
            cells = base[:, None, :] + offsets[None]                              # (B, K, 3)
            valid = ((cells >= 0) & (cells < resolution)).all(dim=-1)
            pos = lo_t + cells.float() * voxel
            d = pos - mu[:, None, :]
            m = torch.einsum('bki,bij,bkj->bk', d, inv_t[idx], d)
            w = alpha_t[idx][:, None] * torch.exp(-0.5 * m) * valid
            flat = (cells[..., 0] * resolution + cells[..., 1]) * resolution + cells[..., 2]
            flat = torch.where(valid, flat, torch.zeros_like(flat)).reshape(-1)
            w = w.reshape(-1)
            density.index_add_(0, flat, w)
            colour.index_add_(0, flat, w[:, None] * rgb_t[idx][:, None, :].expand(-1, len(offsets), -1).reshape(-1, 3))
            done += len(idx)
            if progress:
                progress(done / total)
    colour = colour / density.clamp_min(1e-6)[:, None]
    shape = (resolution,) * 3
    return (density.reshape(shape).cpu().numpy(), colour.reshape(shape + (3,)).cpu().numpy(),
            lo.astype(np.float64), float(voxel))


def sample_colours(colour_grid, points, normals, lo, voxel, offset=0.5):
    """Colour at mesh vertices: trilinear sample of the colour field slightly outside the surface (along the
    normal), where only the visible outer splats contribute; inside, darker interior splats would mix in."""
    from scipy.ndimage import map_coordinates
    coords = ((points + normals * voxel * offset - lo) / voxel).T
    rgb = np.stack([map_coordinates(colour_grid[..., c], coords, order=1, mode='nearest') for c in range(3)], axis=1)
    return np.clip(rgb, 0, 1)


def orient_outward(mesh):
    """Make the face normals point away from the centre. Poisson and marching cubes can wind either way,
    and a mesh with inward normals both lights wrongly and samples its colours from inside the object."""
    d = mesh.triangles_center - mesh.centroid
    if np.einsum('ij,ij->i', mesh.face_normals, d).mean() < 0:
        mesh.invert()
    mesh._cache.clear()


def orient_by_splats(mesh, xyz, samples=3000, radius=0.02):
    """Point every separate piece's faces to the outside, deciding per piece with the splats: outside is the side
    with empty space. A global "away from the centre" rule gets thin open pieces wrong (the top skin of a lamp's
    base lies below the object's centre, so it would face down), and single-sided viewers then show a hole."""
    import trimesh
    from scipy.spatial import cKDTree
    tree = cKDTree(xyz)
    r = radius * float(np.ptp(xyz, axis=0).max())
    parts = mesh.split(only_watertight=False)
    if not parts:
        return mesh
    rng = np.random.default_rng(0)
    fixed = []
    for part in parts:
        part.fix_normals()
        pick = rng.choice(len(part.faces), size=min(samples, len(part.faces)), replace=False)
        centres, normals = part.triangles_center[pick], part.face_normals[pick]
        score = 0
        for c, n, neighbours in zip(centres, normals, tree.query_ball_point(centres, r, workers=-1)):
            if neighbours:
                side = (xyz[neighbours] - c) @ n
                score += int((side > 0).sum()) - int((side < 0).sum())
        if score > 0:  # more splats in front of the faces than behind: they face inward
            part.invert()
        fixed.append(part)
    out = trimesh.util.concatenate(fixed) if len(fixed) > 1 else fixed[0]
    out._cache.clear()
    return out


def apply_material(mesh):
    """Give GLB a matte material. Without one, glTF viewers apply the spec default (metallic), which
    makes vertex-coloured meshes look dark or washed out."""
    from trimesh.visual.material import PBRMaterial
    mesh.visual.material = PBRMaterial(baseColorFactor=[1.0, 1.0, 1.0, 1.0], metallicFactor=0.0,
                                      roughnessFactor=0.9, doubleSided=True)


def linearize_vertex_colors(mesh):
    """glTF COLOR_0 is linear; the splat/PLY colours are display (sRGB) bytes. Without this conversion,
    viewers apply an extra linear-to-sRGB step and the mesh looks pale and washed out."""
    c = np.asarray(mesh.visual.vertex_colors, dtype=np.float64) / 255.0
    rgb = np.where(c[:, :3] <= 0.04045, c[:, :3] / 12.92, ((c[:, :3] + 0.055) / 1.055) ** 2.4)
    mesh.visual.vertex_colors = (np.concatenate([rgb, c[:, 3:4]], axis=1) * 255 + 0.5).astype(np.uint8)


def close_holes(mesh, max_loop=5000):
    """Close the openings the trim leaves (e.g. the inner side of each leg, where Poisson's skin between the
    legs was cut away). Objects from a single image are solid, and game engines and rigging want closed meshes.

    Pure Python on purpose (VTK's filler crashed on these meshes): walk each boundary loop and fan it to its
    centre, wound opposite to the existing edge so the patch faces the same way as its neighbours."""
    import trimesh
    faces = np.asarray(mesh.faces)
    directed = faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2)
    keys = directed[:, 0].astype(np.int64) * (len(mesh.vertices) + 1) + directed[:, 1]
    reverse = directed[:, 1].astype(np.int64) * (len(mesh.vertices) + 1) + directed[:, 0]
    boundary = directed[~np.isin(keys, reverse)]                     # edges with no opposite half-edge
    if len(boundary) == 0:
        return mesh
    # A hole is walked against the existing edges: existing a->b means the hole goes b->a.
    nxt = {}
    for a, b in boundary:
        nxt.setdefault(int(b), []).append(int(a))
    vertices = [np.asarray(mesh.vertices)]
    new_faces = []
    count = len(mesh.vertices)
    while nxt:
        start = next(iter(nxt))
        loop, v = [start], start
        while True:
            targets = nxt.get(v)
            if not targets:
                break
            w = targets.pop()
            if not targets:
                del nxt[v]
            if w == start:
                break
            loop.append(w)
            v = w
            if len(loop) > max_loop:
                break
        if len(loop) < 3 or len(loop) > max_loop:
            continue
        ring = np.asarray(loop)
        centre = np.asarray(mesh.vertices)[ring].mean(axis=0)
        vertices.append(centre[None])
        c = count
        count += 1
        for i in range(len(ring)):
            a, b = ring[i], ring[(i + 1) % len(ring)]
            new_faces.append((a, b, c))
    if not new_faces:
        return mesh
    out = trimesh.Trimesh(np.concatenate(vertices), np.concatenate([faces, np.asarray(new_faces)]), process=False)
    out.update_faces(out.nondegenerate_faces())
    out.remove_unreferenced_vertices()
    return out


def poisson_mesh(splats, resolution=256, depth=None, min_alpha=0.05, trim=0.01, smooth=3, faces=200_000,
                 progress=None):
    """Screened-Poisson surface from the Gaussian centres, with normals from each Gaussian's smallest
    axis. This keeps the face, eyes and thin antenna far better than an opacity density field, and the
    colours come from the nearest splats instead of a blurred voxel average.

    min_alpha is low on purpose: the sides Idol invents (the back of an object) are made of fainter splats,
    and dropping them leaves holes there. Poisson closes every surface, so afterwards triangles farther than
    `trim` x the object size from any splat are removed: that cuts the skin Poisson stretches across real
    gaps (between legs, under handles) without opening holes where there are splats."""
    import open3d as o3d
    from scipy.spatial import cKDTree

    stage = progress or (lambda *_: None)
    if depth is None:
        depth = {128: 7, 192: 8, 256: 9, 384: 10, 512: 10}.get(int(resolution), 9)

    keep = splats.alpha >= min_alpha
    if keep.sum() < 100:
        keep = np.ones(splats.count, dtype=bool)
    xyz, alpha, scale, rot, rgb = (splats.xyz[keep], splats.alpha[keep], splats.scale[keep],
                                   splats.rot[keep], splats.rgb[keep])
    stage('surface', 0.0)

    R = quaternion_to_matrix(rot)
    axis = np.argmin(scale, axis=1)
    normals = R[np.arange(len(xyz)), :, axis]
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-9)
    centre = np.median(xyz, axis=0)
    normals[np.einsum('ij,ij->i', normals, xyz - centre) < 0] *= -1

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    pcd.normals = o3d.utility.Vector3dVector(normals.astype(np.float64))
    surface, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=int(depth), scale=1.1, linear_fit=True)
    if len(surface.vertices) == 0:
        raise ValueError('Poisson reconstruction produced an empty mesh')

    import trimesh
    mesh = trimesh.Trimesh(np.asarray(surface.vertices), np.asarray(surface.triangles), process=True)
    tree = cKDTree(xyz)
    if trim > 0:
        size = float(np.ptp(xyz, axis=0).max())
        far = tree.query(mesh.triangles_center, workers=-1)[0] > trim * size
        if far.any() and not far.all():
            mesh.update_faces(~far)
            mesh.remove_unreferenced_vertices()
    parts = mesh.split(only_watertight=False)
    if len(parts) > 1:
        largest = max(len(p.faces) for p in parts)
        parts = [p for p in parts if len(p.faces) >= max(0.01 * largest, 50)]
        mesh = trimesh.util.concatenate(parts)
    if trim > 0:
        mesh.merge_vertices()
        mesh = close_holes(mesh)

    if smooth > 0:
        stage('smooth', 0.0)
        trimesh.smoothing.filter_taubin(mesh, lamb=0.5, nu=-0.53, iterations=int(smooth))
    if faces and len(mesh.faces) > faces:
        stage('simplify', 0.0)
        try:
            mesh = mesh.simplify_quadric_decimation(face_count=int(faces))
        except Exception as error:  # fast-simplification missing: keep the full mesh
            print(f'simplify skipped: {error}', file=sys.stderr, flush=True)
    # Decimation can leave crumbs; keep only real pieces.
    parts = mesh.split(only_watertight=False)
    if len(parts) > 1:
        largest = max(len(p.faces) for p in parts)
        mesh = trimesh.util.concatenate([p for p in parts if len(p.faces) >= max(0.01 * largest, 50)])

    mesh = orient_by_splats(mesh, xyz)
    stage('colour', 0.0)
    k = min(12, len(xyz))
    offset = COLOUR_OFFSET * float(np.ptp(xyz, axis=0).max())
    distances, nearest = tree.query(np.asarray(mesh.vertices) + np.asarray(mesh.vertex_normals) * offset, k=k,
                                    workers=-1)
    if k == 1:
        colours = rgb[nearest]
    else:
        weights = alpha[nearest] / (distances + 1e-5) ** 2
        colours = (rgb[nearest] * weights[..., None]).sum(axis=1) / weights.sum(axis=1, keepdims=True)
    mesh.visual.vertex_colors = (np.concatenate([np.clip(colours, 0, 1), np.ones((len(colours), 1))], axis=1)
                                 * 255 + 0.5).astype(np.uint8)
    return mesh


def make_mesh(splats, resolution=256, threshold=0.5, smooth=3, faces=200_000, min_component=0.01,
              method='poisson', min_alpha=0.05, trim=0.01, progress=None):
    if method == 'poisson':
        try:
            return poisson_mesh(splats, resolution=resolution, min_alpha=min_alpha, trim=trim, smooth=smooth,
                                faces=faces, progress=progress)
        except ImportError:
            print('open3d is not installed; using the density-field mesh (run setup to install it)',
                  file=sys.stderr, flush=True)
        except Exception as error:
            print(f'Poisson reconstruction failed ({error}); using the density-field mesh', file=sys.stderr,
                  flush=True)

    import trimesh
    from skimage.measure import marching_cubes

    stage = progress or (lambda *_: None)
    stage('density', 0.0)
    density, colour, lo, voxel = density_grids(splats, resolution, progress=lambda f: stage('density', f))
    # A slight blur turns the overlapping flat splats into a smooth field (less lumpy surface).
    from scipy.ndimage import gaussian_filter
    density = gaussian_filter(density, sigma=0.9 * resolution / 256)
    if density.max() <= threshold:
        raise ValueError(f'nothing above the surface threshold {threshold} (max density {density.max():.3f}); '
                         'lower the threshold')
    stage('surface', 0.0)
    verts, tris, _, _ = marching_cubes(density, level=threshold, spacing=(voxel,) * 3)
    verts = verts + lo
    mesh = trimesh.Trimesh(verts, tris, process=True)

    # Drop floaters: keep components with at least min_component of the faces.
    parts = mesh.split(only_watertight=False)
    if len(parts) > 1:
        largest = max(len(p.faces) for p in parts)
        parts = [p for p in parts if len(p.faces) >= max(min_component * largest, 50)]
        mesh = trimesh.util.concatenate(parts)

    if smooth > 0:
        stage('smooth', 0.0)
        trimesh.smoothing.filter_taubin(mesh, lamb=0.5, nu=-0.53, iterations=int(smooth))

    if faces and len(mesh.faces) > faces:
        stage('simplify', 0.0)
        try:
            mesh = mesh.simplify_quadric_decimation(face_count=int(faces))
        except Exception as error:  # fast-simplification missing: keep the full mesh
            print(f'simplify skipped: {error}', file=sys.stderr, flush=True)

    mesh.fix_normals()
    orient_outward(mesh)
    rgb = sample_colours(colour, mesh.vertices, mesh.vertex_normals, lo, voxel)
    mesh.visual.vertex_colors = (np.concatenate([rgb, np.ones((len(rgb), 1))], axis=1) * 255 + 0.5).astype(np.uint8)
    return mesh


# ---------------------------------------------------------------------------------------------------------------
# Game-ready mesh: low poly + UVs + baked texture

def splat_colours(tree, xyz, rgb, alpha, points, k=12):
    """Colour at arbitrary points from the nearest splats (opacity / distance^2 weighted), in display sRGB."""
    k = min(k, len(xyz))
    distances, nearest = tree.query(points, k=k, workers=-1)
    if k == 1:
        return rgb[nearest]
    weights = alpha[nearest] / (distances + 1e-5) ** 2
    return (rgb[nearest] * weights[..., None]).sum(axis=1) / weights.sum(axis=1, keepdims=True)


def rasterize_uv(uvs, faces, size):
    """Every texel centre covered by a UV triangle: its (row, col), triangle and barycentrics.
    UVs are OpenGL style (v up); image row 0 is the top (v = 1)."""
    P = np.stack([uvs[:, 0] * size, (1.0 - uvs[:, 1]) * size], axis=1)
    tri = P[faces]                                                         # (F, 3, 2)
    x0s = np.floor(tri[..., 0].min(axis=1) - 0.5).astype(int).clip(0, size - 1)
    x1s = np.ceil(tri[..., 0].max(axis=1) - 0.5).astype(int).clip(0, size - 1)
    y0s = np.floor(tri[..., 1].min(axis=1) - 0.5).astype(int).clip(0, size - 1)
    y1s = np.ceil(tri[..., 1].max(axis=1) - 0.5).astype(int).clip(0, size - 1)
    rows, cols, fids, bars = [], [], [], []
    for f in range(len(faces)):
        xs, ys = np.meshgrid(np.arange(x0s[f], x1s[f] + 1), np.arange(y0s[f], y1s[f] + 1))
        xs, ys = xs.ravel(), ys.ravel()
        px, py = xs + 0.5, ys + 0.5
        (ax, ay), (bx, by), (cx, cy) = tri[f]
        det = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
        if abs(det) < 1e-12:
            continue
        l0 = ((by - cy) * (px - cx) + (cx - bx) * (py - cy)) / det
        l1 = ((cy - ay) * (px - cx) + (ax - cx) * (py - cy)) / det
        l2 = 1.0 - l0 - l1
        inside = (l0 >= -1e-6) & (l1 >= -1e-6) & (l2 >= -1e-6)
        if not inside.any():
            continue
        rows.append(ys[inside])
        cols.append(xs[inside])
        fids.append(np.full(int(inside.sum()), f))
        bars.append(np.stack([l0, l1, l2], axis=1)[inside])
    if not rows:
        raise ValueError('UV rasterization produced no texels')
    return np.concatenate(rows), np.concatenate(cols), np.concatenate(fids), np.concatenate(bars)


def game_mesh(splats, faces=20_000, texture_size=2048, resolution=256, smooth=3, min_alpha=0.05, trim=0.01,
              progress=None):
    """Low-poly, UV-unwrapped mesh with the splat colours baked into a texture.
    Returns (vertices, faces, normals, uvs, texture image) in the PLY frame."""
    import trimesh
    import xatlas
    from PIL import Image
    from scipy.ndimage import distance_transform_edt
    from scipy.spatial import cKDTree

    stage = progress or (lambda *_: None)
    faces = max(int(faces or 20_000), 500)
    high = poisson_mesh(splats, resolution=resolution, min_alpha=min_alpha, trim=trim, smooth=smooth,
                        faces=max(faces * 4, 100_000), progress=progress)
    stage('simplify', 0.0)
    low = high.simplify_quadric_decimation(face_count=faces) if len(high.faces) > faces else high.copy()
    parts = low.split(only_watertight=False)
    if len(parts) > 1:
        largest = max(len(p.faces) for p in parts)
        low = trimesh.util.concatenate([p for p in parts if len(p.faces) >= max(0.01 * largest, 20)])
    low.merge_vertices()
    keep_mask = splats.alpha >= min_alpha
    low = orient_by_splats(low, splats.xyz[keep_mask])
    normals = np.asarray(low.vertex_normals, dtype=np.float64)

    stage('unwrap', 0.0)
    vmap, tris, uvs = xatlas.parametrize(np.asarray(low.vertices, np.float32), np.asarray(low.faces, np.uint32))
    verts, norms = np.asarray(low.vertices, dtype=np.float64)[vmap], normals[vmap]

    stage('bake', 0.0)
    keep = splats.alpha >= min_alpha
    xyz, rgb, alpha = splats.xyz[keep], splats.rgb[keep], splats.alpha[keep]
    tree = cKDTree(xyz)
    size = int(texture_size)
    rows, cols, fids, bars = rasterize_uv(uvs, tris, size)
    points = np.einsum('tk,tkj->tj', bars, verts[tris[fids]])
    # The low-poly surface cuts inside curved areas (a flat triangle across a dome), where the nearest splats
    # are the darker interior ones. Lift each texel along its normal onto the full-detail surface first.
    texel_normals = np.einsum('tk,tkj->tj', bars, norms[tris[fids]])
    texel_normals /= np.maximum(np.linalg.norm(texel_normals, axis=1, keepdims=True), 1e-9)
    high_vertices = np.asarray(high.vertices)
    high_tree = cKDTree(high_vertices)
    for start in range(0, len(points), 1_000_000):
        sl = slice(start, start + 1_000_000)
        idx = high_tree.query(points[sl], workers=-1)[1]
        lift = np.einsum('ij,ij->i', high_vertices[idx] - points[sl], texel_normals[sl])
        points[sl] += texel_normals[sl] * (lift[:, None] + COLOUR_OFFSET * float(np.ptp(xyz, axis=0).max()))
    image = np.zeros((size, size, 3), dtype=np.float32)
    chunk = 1_000_000
    for start in range(0, len(points), chunk):
        sl = slice(start, start + chunk)
        image[rows[sl], cols[sl]] = splat_colours(tree, xyz, rgb, alpha, points[sl])
        stage('bake', min(1.0, (start + chunk) / len(points)))
    mask = np.zeros((size, size), dtype=bool)
    mask[rows, cols] = True
    # Fill every empty texel with the nearest baked one: no dark seams at island edges or in mipmaps.
    nearest = distance_transform_edt(~mask, return_distances=False, return_indices=True)
    image = image[nearest[0], nearest[1]]
    texture = Image.fromarray((np.clip(image, 0, 1) * 255 + 0.5).astype(np.uint8), 'RGB')
    return verts, tris.astype(np.int64), norms, uvs.astype(np.float64), texture


def orient_arrays(v, n, height=None):
    """The y-up transform of orient_mesh, for vertices and normals."""
    v, n = upright(v), upright(n)
    for a in (0, 2):
        v[:, a] -= (v[:, a].min() + v[:, a].max()) / 2
    v[:, 1] -= v[:, 1].min()
    if height:
        v *= float(height) / max(v[:, 1].max(), 1e-9)
    return v, n


def write_game_glb(splats, path, faces=20_000, texture_size=2048, height=None, progress=None, **kw):
    import trimesh
    from trimesh.visual.material import PBRMaterial
    from trimesh.visual.texture import TextureVisuals
    verts, tris, norms, uvs, texture = game_mesh(splats, faces=faces, texture_size=texture_size, progress=progress,
                                                 **kw)
    verts, norms = orient_arrays(verts, norms, height)
    # The texture is display sRGB, which is what glTF expects for baseColorTexture (no conversion needed).
    material = PBRMaterial(baseColorTexture=texture, baseColorFactor=[1.0, 1.0, 1.0, 1.0], metallicFactor=0.0,
                           roughnessFactor=0.9)
    mesh = trimesh.Trimesh(verts, tris, vertex_normals=norms, visual=TextureVisuals(uv=uvs, material=material),
                           process=False)
    mesh.export(path, file_type='glb', include_normals=True)
    return mesh


def orient_mesh(mesh, fmt, height=None):
    """y-up for glb / obj / mesh-ply, z-up for stl; optionally scale so the object is `height` units tall."""
    v = upright(mesh.vertices)  # y-up
    if fmt == 'stl':
        v = np.stack([v[:, 0], -v[:, 2], v[:, 1]], axis=1)  # y-up -> z-up
        up = 2
    else:
        up = 1
    # Centre horizontally and stand on the ground plane.
    horizontal = [a for a in range(3) if a != up]
    for a in horizontal:
        v[:, a] -= (v[:, a].min() + v[:, a].max()) / 2
    v[:, up] -= v[:, up].min()
    if height:
        v *= float(height) / max(v[:, up].max(), 1e-9)
    mesh.vertices = v
    # The y/z swaps above keep handedness (two sign flips / a rotation), so the winding stays correct.
    # Drop cached normals: they still describe the old orientation.
    mesh._cache.clear()
    _ = mesh.vertex_normals  # recompute and cache, so exporters write the new normals
    return mesh


# ---------------------------------------------------------------------------------------------------------------

def export(ply_path, fmt, out_path=None, resolution=256, threshold=0.5, smooth=3, faces=200_000, height=None,
           method='poisson', min_alpha=0.05, trim=0.01, texture_size=2048, game_faces=20_000, progress=None):
    """Export `ply_path` to `fmt`. Returns a dict with the output path and some stats."""
    fmt = fmt.lower()
    if fmt not in FORMATS:
        raise ValueError(f'unknown format {fmt}; use one of: {", ".join(FORMATS)}')
    out_path = out_path or default_path(ply_path, fmt)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    started = time.time()
    info = {'format': fmt, 'path': out_path}
    if fmt == 'ply':
        if os.path.abspath(out_path) != os.path.abspath(ply_path):
            shutil.copyfile(ply_path, out_path)
    else:
        splats = read_ply(ply_path)
        info['splats'] = splats.count
        if fmt == 'game-glb':
            size = int(texture_size or 2048)
            mesh = write_game_glb(splats, out_path, faces=int(game_faces or 20_000), texture_size=size,
                                  height=height, progress=progress, resolution=int(resolution), smooth=int(smooth),
                                  min_alpha=min_alpha, trim=trim)
            info.update(vertices=int(len(mesh.vertices)), faces=int(len(mesh.faces)), texture_size=size,
                        size=[float(x) for x in mesh.extents])
        elif fmt == 'spz':
            write_spz(splats, out_path)
        elif fmt == 'splat':
            write_splat(splats, out_path)
        elif fmt == 'points':
            write_points(splats, out_path)
        else:
            mesh = make_mesh(splats, resolution=int(resolution), threshold=float(threshold), smooth=int(smooth),
                             faces=int(faces) if faces else 0, method=method, min_alpha=min_alpha, trim=trim,
                             progress=progress)
            mesh = orient_mesh(mesh, fmt, height)
            if fmt == 'glb':
                linearize_vertex_colors(mesh)
                apply_material(mesh)
                _ = mesh.vertex_normals
                mesh.export(out_path, file_type='glb')
            elif fmt == 'obj':
                mesh.export(out_path, file_type='obj', include_color=True)
            elif fmt == 'stl':
                mesh.export(out_path, file_type='stl')
            else:
                mesh.export(out_path, file_type='ply', encoding='binary')
            info.update(vertices=int(len(mesh.vertices)), faces=int(len(mesh.faces)),
                        watertight=bool(mesh.is_watertight), size=[float(x) for x in mesh.extents])
    info['bytes'] = os.path.getsize(out_path)
    info['seconds'] = round(time.time() - started, 2)
    return info


if __name__ == '__main__':
    import argparse
    import json

    parser = argparse.ArgumentParser(description='Export an Idol 3DGS .ply')
    parser.add_argument('ply')
    parser.add_argument('out')
    parser.add_argument('--format')
    parser.add_argument('--resolution', type=int, default=256)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--smooth', type=int, default=3)
    parser.add_argument('--faces', type=int, default=200_000)
    parser.add_argument('--height', type=float)
    parser.add_argument('--method', choices=['poisson', 'density'], default='poisson')
    parser.add_argument('--min-alpha', type=float, default=0.05)
    parser.add_argument('--trim', type=float, default=0.01)
    parser.add_argument('--texture-size', type=int, default=2048)
    parser.add_argument('--game-faces', type=int, default=20_000)
    a = parser.parse_args()
    fmt = a.format or next((f for f, ext in EXTENSIONS.items() if a.out.lower().endswith(ext) and f != 'ply'),
                           os.path.splitext(a.out)[1][1:])
    print(json.dumps(export(a.ply, fmt, a.out, resolution=a.resolution, threshold=a.threshold, smooth=a.smooth,
                            faces=a.faces, height=a.height, method=a.method, min_alpha=a.min_alpha, trim=a.trim,
                            texture_size=a.texture_size, game_faces=a.game_faces,
                            progress=lambda s, f: None)))
