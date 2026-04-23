from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class MeshData:
    vertices: np.ndarray
    faces: np.ndarray
    uvs: Optional[np.ndarray] = None
    colors: Optional[np.ndarray] = None


class SurfaceEngine:
    """Meshing and texturing stage for the 2D-to-3D pipeline.

    The implementation intentionally depends only on numpy and optional cv2 so
    it can run with the current project requirements. It can build a mesh from:
    - depth maps, when they are passed directly or attached to sparse_data
    - sparse 3D points, by projecting points to a dominant plane and gridding
    """

    def __init__(
        self,
        output_dir: str = "data/outputs",
        mesh_name: str = "surface_mesh.obj",
        grid_size: int = 160,
        depth_stride: int = 4,
        max_texture_size: int = 2048,
        camera_matrix: Optional[np.ndarray] = None,
        max_edge_length: Optional[float] = None,
    ) -> None:
        project_root = Path(__file__).resolve().parents[2]
        output_path = Path(output_dir)
        self.output_dir = output_path if output_path.is_absolute() else project_root / output_path
        self.mesh_name = mesh_name
        self.grid_size = max(4, int(grid_size))
        self.depth_stride = max(1, int(depth_stride))
        self.max_texture_size = int(max_texture_size)
        self.camera_matrix = None if camera_matrix is None else np.asarray(camera_matrix, dtype=np.float32)
        self.max_edge_length = max_edge_length

    def generate_dense_mesh(
        self,
        sparse_data: Any,
        images: Sequence[np.ndarray],
        output_dir: Optional[str] = None,
        mesh_name: Optional[str] = None,
        depth_maps: Optional[Sequence[np.ndarray]] = None,
    ) -> str:
        """Generate a textured OBJ mesh and return the OBJ file path."""
        target_dir = self._resolve_output_dir(output_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

        obj_path = target_dir / (mesh_name or self.mesh_name)
        if obj_path.suffix.lower() != ".obj":
            obj_path = obj_path.with_suffix(".obj")

        extracted_depth_maps = self._normalize_depth_maps(
            depth_maps if depth_maps is not None else self._extract_depth_maps(sparse_data)
        )
        if extracted_depth_maps is not None and len(extracted_depth_maps) > 0:
            mesh = self._mesh_from_depth_map(extracted_depth_maps[0], self._first_image(images), sparse_data)
        else:
            points, colors = self._extract_point_cloud(sparse_data)
            mesh = self._mesh_from_points(points, colors)

        texture_path = self._write_texture(self._first_image(images), obj_path)
        self._write_obj(mesh, obj_path, texture_path)
        return str(obj_path)

    def generate_textured_mesh(
        self,
        sparse_data: Any,
        images: Sequence[np.ndarray],
        output_dir: Optional[str] = None,
        mesh_name: Optional[str] = None,
        depth_maps: Optional[Sequence[np.ndarray]] = None,
    ) -> str:
        """Alias for callers that name stage 7 explicitly."""
        return self.generate_dense_mesh(
            sparse_data=sparse_data,
            images=images,
            output_dir=output_dir,
            mesh_name=mesh_name,
            depth_maps=depth_maps,
        )

    def visualize(self, mesh_path: str) -> None:
        """Print a minimal preview summary without launching external viewers."""
        path = Path(mesh_path)
        if not path.exists():
            raise FileNotFoundError(f"Mesh file not found: {path}")
        print(f"Mesh ready: {path.resolve()}")

    def _resolve_output_dir(self, output_dir: Optional[str]) -> Path:
        if output_dir is None:
            return self.output_dir
        path = Path(output_dir)
        if path.is_absolute():
            return path
        project_root = Path(__file__).resolve().parents[2]
        return project_root / path

    @staticmethod
    def _first_image(images: Sequence[np.ndarray]) -> Optional[np.ndarray]:
        if images is None or len(images) == 0:
            return None
        return images[0]

    @staticmethod
    def _field(data: Any, names: Sequence[str], default: Any = None) -> Any:
        if data is None:
            return default
        if isinstance(data, dict):
            for name in names:
                if name in data:
                    return data[name]
            return default
        for name in names:
            if hasattr(data, name):
                return getattr(data, name)
        return default

    def _extract_point_cloud(self, sparse_data: Any) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        points = self._field(sparse_data, ("points_3d", "points", "vertices"))
        if points is None:
            if isinstance(sparse_data, np.ndarray):
                points = sparse_data
            else:
                raise ValueError("sparse_data must contain points_3d, points, or vertices")

        points_arr = np.asarray(points, dtype=np.float32)
        if points_arr.ndim != 2 or points_arr.shape[1] != 3:
            raise ValueError("Point cloud must have shape (N, 3)")

        finite_mask = np.isfinite(points_arr).all(axis=1)
        points_arr = points_arr[finite_mask]
        if len(points_arr) < 3:
            raise ValueError("At least 3 valid 3D points are required for meshing")

        colors = self._field(sparse_data, ("point_colors", "colors", "vertex_colors"))
        color_arr = None
        if colors is not None:
            color_arr = np.asarray(colors)
            if color_arr.ndim == 2 and color_arr.shape[0] == len(finite_mask):
                color_arr = color_arr[finite_mask]
            if color_arr.ndim != 2 or color_arr.shape[1] < 3 or len(color_arr) != len(points_arr):
                color_arr = None
            else:
                color_arr = self._normalize_colors(color_arr[:, :3])

        return points_arr, color_arr

    def _extract_depth_maps(self, sparse_data: Any) -> Optional[Sequence[np.ndarray]]:
        depth_maps = self._field(sparse_data, ("depth_maps", "depth_map"))
        return self._normalize_depth_maps(depth_maps)

    @staticmethod
    def _normalize_depth_maps(depth_maps: Any) -> Optional[List[np.ndarray]]:
        if depth_maps is None:
            return None
        if isinstance(depth_maps, np.ndarray) and depth_maps.ndim == 2:
            return [depth_maps]
        return list(depth_maps)

    @staticmethod
    def _normalize_colors(colors: np.ndarray) -> np.ndarray:
        colors = np.asarray(colors, dtype=np.float32)
        if colors.size == 0:
            return colors.reshape(0, 3)
        if np.nanmax(colors) > 1.0:
            colors = colors / 255.0
        return np.clip(colors, 0.0, 1.0)

    def _mesh_from_points(self, points: np.ndarray, colors: Optional[np.ndarray]) -> MeshData:
        coords = self._project_points_to_plane(points)
        min_xy = coords.min(axis=0)
        max_xy = coords.max(axis=0)
        span = np.maximum(max_xy - min_xy, 1e-6)

        normalized = (coords - min_xy) / span
        grid_xy = np.clip((normalized * (self.grid_size - 1)).astype(np.int32), 0, self.grid_size - 1)

        cells: Dict[Tuple[int, int], List[int]] = {}
        for index, (gx, gy) in enumerate(grid_xy):
            cells.setdefault((int(gx), int(gy)), []).append(index)

        vertex_lookup: Dict[Tuple[int, int], int] = {}
        vertices: List[np.ndarray] = []
        uvs: List[Tuple[float, float]] = []
        vertex_colors: List[np.ndarray] = []

        for cell, indices in sorted(cells.items()):
            selected_points = points[indices]
            vertex_lookup[cell] = len(vertices)
            vertices.append(selected_points.mean(axis=0))
            gx, gy = cell
            uvs.append((gx / (self.grid_size - 1), 1.0 - gy / (self.grid_size - 1)))
            if colors is not None:
                vertex_colors.append(colors[indices].mean(axis=0))

        vertex_arr = np.asarray(vertices, dtype=np.float32)
        uv_arr = np.asarray(uvs, dtype=np.float32)
        color_arr = np.asarray(vertex_colors, dtype=np.float32) if vertex_colors else None
        faces = self._grid_faces(vertex_lookup, vertex_arr)

        if len(faces) == 0:
            faces = self._fan_faces(vertex_arr)

        return MeshData(vertices=vertex_arr, faces=faces, uvs=uv_arr, colors=color_arr)

    def _mesh_from_depth_map(
        self,
        depth_map: np.ndarray,
        image: Optional[np.ndarray],
        sparse_data: Any,
    ) -> MeshData:
        depth = np.asarray(depth_map, dtype=np.float32)
        if depth.ndim != 2:
            raise ValueError("Depth map must have shape (H, W)")

        height, width = depth.shape
        K = self._camera_matrix_for_depth(width, height, sparse_data)
        fx = float(K[0, 0]) if K[0, 0] != 0 else float(max(width, height))
        fy = float(K[1, 1]) if K[1, 1] != 0 else fx
        cx = float(K[0, 2])
        cy = float(K[1, 2])

        vertex_grid: Dict[Tuple[int, int], int] = {}
        vertices: List[Tuple[float, float, float]] = []
        uvs: List[Tuple[float, float]] = []
        colors: List[Tuple[float, float, float]] = []

        rows = range(0, height, self.depth_stride)
        cols = range(0, width, self.depth_stride)
        for y in rows:
            for x in cols:
                z = float(depth[y, x])
                if not np.isfinite(z) or z <= 0.0:
                    continue
                px = (x - cx) * z / fx
                py = (y - cy) * z / fy
                vertex_grid[(y, x)] = len(vertices)
                vertices.append((px, py, z))
                uvs.append((x / max(width - 1, 1), 1.0 - y / max(height - 1, 1)))
                if image is not None:
                    colors.append(self._sample_image_color(image, x, y, width, height))

        vertex_arr = np.asarray(vertices, dtype=np.float32)
        if len(vertex_arr) < 3:
            raise ValueError("Depth map does not contain enough valid depth values")

        faces: List[Tuple[int, int, int]] = []
        for y in range(0, height - self.depth_stride, self.depth_stride):
            for x in range(0, width - self.depth_stride, self.depth_stride):
                a = vertex_grid.get((y, x))
                b = vertex_grid.get((y, x + self.depth_stride))
                c = vertex_grid.get((y + self.depth_stride, x))
                d = vertex_grid.get((y + self.depth_stride, x + self.depth_stride))
                if a is None or b is None or c is None or d is None:
                    continue
                faces.extend(self._quad_faces(vertex_arr, a, b, c, d))

        if len(faces) == 0:
            faces_arr = self._fan_faces(vertex_arr)
        else:
            faces_arr = np.asarray(faces, dtype=np.int32)

        color_arr = np.asarray(colors, dtype=np.float32) if colors else None
        return MeshData(
            vertices=vertex_arr,
            faces=faces_arr,
            uvs=np.asarray(uvs, dtype=np.float32),
            colors=color_arr,
        )

    def _camera_matrix_for_depth(self, width: int, height: int, sparse_data: Any) -> np.ndarray:
        matrix = self.camera_matrix
        if matrix is None:
            matrix = self._field(sparse_data, ("camera_matrix", "calibration_matrix", "K"))
        if matrix is not None:
            matrix_arr = np.asarray(matrix, dtype=np.float32)
            if matrix_arr.shape == (3, 3):
                return matrix_arr

        focal = float(max(width, height))
        return np.array(
            [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )

    @staticmethod
    def _project_points_to_plane(points: np.ndarray) -> np.ndarray:
        centered = points - points.mean(axis=0, keepdims=True)
        try:
            _, _, vh = np.linalg.svd(centered, full_matrices=False)
            axes = vh[:2].T
            return centered @ axes
        except np.linalg.LinAlgError:
            return points[:, :2]

    def _grid_faces(self, vertex_lookup: Dict[Tuple[int, int], int], vertices: np.ndarray) -> np.ndarray:
        faces: List[Tuple[int, int, int]] = []
        for gx in range(self.grid_size - 1):
            for gy in range(self.grid_size - 1):
                a = vertex_lookup.get((gx, gy))
                b = vertex_lookup.get((gx + 1, gy))
                c = vertex_lookup.get((gx, gy + 1))
                d = vertex_lookup.get((gx + 1, gy + 1))
                if a is None or b is None or c is None or d is None:
                    continue
                faces.extend(self._quad_faces(vertices, a, b, c, d))
        return np.asarray(faces, dtype=np.int32)

    def _quad_faces(
        self,
        vertices: np.ndarray,
        a: int,
        b: int,
        c: int,
        d: int,
    ) -> List[Tuple[int, int, int]]:
        if self._quad_has_long_edge(vertices, (a, b, c, d)):
            return []
        diag_ac = np.linalg.norm(vertices[a] - vertices[d])
        diag_bc = np.linalg.norm(vertices[b] - vertices[c])
        if diag_ac <= diag_bc:
            return [(a, b, d), (a, d, c)]
        return [(a, b, c), (b, d, c)]

    def _quad_has_long_edge(self, vertices: np.ndarray, indices: Tuple[int, int, int, int]) -> bool:
        if self.max_edge_length is None:
            return False
        a, b, c, d = indices
        edges = (
            np.linalg.norm(vertices[a] - vertices[b]),
            np.linalg.norm(vertices[b] - vertices[d]),
            np.linalg.norm(vertices[d] - vertices[c]),
            np.linalg.norm(vertices[c] - vertices[a]),
        )
        return max(edges) > self.max_edge_length

    @staticmethod
    def _fan_faces(vertices: np.ndarray) -> np.ndarray:
        if len(vertices) < 3:
            return np.empty((0, 3), dtype=np.int32)
        center_index = int(np.argmin(np.linalg.norm(vertices - vertices.mean(axis=0), axis=1)))
        other_indices = [idx for idx in range(len(vertices)) if idx != center_index]
        if len(other_indices) < 2:
            return np.empty((0, 3), dtype=np.int32)

        projected = SurfaceEngine._project_points_to_plane(vertices[other_indices])
        angles = np.arctan2(projected[:, 1], projected[:, 0])
        ordered = [other_indices[idx] for idx in np.argsort(angles)]
        faces = [(center_index, ordered[idx], ordered[(idx + 1) % len(ordered)]) for idx in range(len(ordered))]
        return np.asarray(faces, dtype=np.int32)

    @staticmethod
    def _sample_image_color(
        image: np.ndarray,
        depth_x: int,
        depth_y: int,
        depth_width: int,
        depth_height: int,
    ) -> Tuple[float, float, float]:
        if image.ndim == 2:
            value = float(image[
                min(int(depth_y * image.shape[0] / max(depth_height, 1)), image.shape[0] - 1),
                min(int(depth_x * image.shape[1] / max(depth_width, 1)), image.shape[1] - 1),
            ]) / 255.0
            return value, value, value

        y = min(int(depth_y * image.shape[0] / max(depth_height, 1)), image.shape[0] - 1)
        x = min(int(depth_x * image.shape[1] / max(depth_width, 1)), image.shape[1] - 1)
        bgr = image[y, x, :3].astype(np.float32) / 255.0
        return float(bgr[2]), float(bgr[1]), float(bgr[0])

    def _write_texture(self, image: Optional[np.ndarray], obj_path: Path) -> Optional[Path]:
        if image is None:
            return None

        try:
            import cv2 as cv
        except ModuleNotFoundError:
            return None

        texture_path = obj_path.with_name(f"{obj_path.stem}_texture.jpg")
        texture = image.copy()
        if self.max_texture_size > 0:
            height, width = texture.shape[:2]
            scale = min(1.0, self.max_texture_size / max(height, width))
            if scale < 1.0:
                texture = cv.resize(
                    texture,
                    (max(1, int(width * scale)), max(1, int(height * scale))),
                    interpolation=cv.INTER_AREA,
                )
        if not cv.imwrite(str(texture_path), texture):
            return None
        return texture_path

    def _write_obj(self, mesh: MeshData, obj_path: Path, texture_path: Optional[Path]) -> None:
        mtl_path = obj_path.with_suffix(".mtl")
        has_uvs = mesh.uvs is not None and len(mesh.uvs) == len(mesh.vertices)
        has_colors = mesh.colors is not None and len(mesh.colors) == len(mesh.vertices)

        if texture_path is not None:
            self._write_mtl(mtl_path, texture_path)

        with obj_path.open("w", encoding="utf-8") as obj:
            obj.write("# Generated by SurfaceEngine\n")
            if texture_path is not None:
                obj.write(f"mtllib {mtl_path.name}\n")
                obj.write("usemtl material_0\n")

            for index, vertex in enumerate(mesh.vertices):
                if has_colors:
                    color = mesh.colors[index]
                    obj.write(
                        f"v {vertex[0]:.8f} {vertex[1]:.8f} {vertex[2]:.8f} "
                        f"{color[0]:.6f} {color[1]:.6f} {color[2]:.6f}\n"
                    )
                else:
                    obj.write(f"v {vertex[0]:.8f} {vertex[1]:.8f} {vertex[2]:.8f}\n")

            if has_uvs:
                for uv in mesh.uvs:
                    obj.write(f"vt {uv[0]:.8f} {uv[1]:.8f}\n")

            for face in mesh.faces:
                a, b, c = (int(face[0]) + 1, int(face[1]) + 1, int(face[2]) + 1)
                if has_uvs:
                    obj.write(f"f {a}/{a} {b}/{b} {c}/{c}\n")
                else:
                    obj.write(f"f {a} {b} {c}\n")

    @staticmethod
    def _write_mtl(mtl_path: Path, texture_path: Path) -> None:
        with mtl_path.open("w", encoding="utf-8") as mtl:
            mtl.write("newmtl material_0\n")
            mtl.write("Ka 1.000000 1.000000 1.000000\n")
            mtl.write("Kd 1.000000 1.000000 1.000000\n")
            mtl.write("Ks 0.000000 0.000000 0.000000\n")
            mtl.write("d 1.000000\n")
            mtl.write("illum 2\n")
            mtl.write(f"map_Kd {texture_path.name}\n")


Surface_Engine = SurfaceEngine
