from __future__ import annotations

import importlib
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from utils.config import SparseCloud, VerifiedMatches


class GeometryEngine:
    def __init__(
        self,
        calibration_matrix: np.ndarray,
        baseline: float = 0.1,
        num_disparities: int = 128,
        block_size: int = 5,
        output_dir: Optional[str] = None,
    ) -> None:
        if calibration_matrix.shape != (3, 3):
            raise ValueError("calibration_matrix must be a 3x3 matrix")

        self.K = calibration_matrix.astype(np.float32)
        self.baseline = float(baseline)
        self.num_disparities = self._normalize_num_disparities(num_disparities)
        self.block_size = self._normalize_block_size(block_size)
        self.output_dir = Path(output_dir) if output_dir else None

    @staticmethod
    def _normalize_num_disparities(num_disparities: int) -> int:
        value = max(16, int(num_disparities))
        return value + (16 - value % 16) % 16

    @staticmethod
    def _normalize_block_size(block_size: int) -> int:
        value = max(3, int(block_size))
        return value if value % 2 == 1 else value + 1

    @staticmethod
    def _to_grayscale(cv2_module, image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            return image
        return cv2_module.cvtColor(image, cv2_module.COLOR_BGR2GRAY)

    def _load_cv2(self):
        try:
            return importlib.import_module("cv2")
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "OpenCV (cv2) is required for depth estimation. Install it with: pip install opencv-python"
            ) from exc

    def estimate_depth_maps(self, images: List[np.ndarray]) -> List[np.ndarray]:
        if len(images) < 2:
            raise ValueError("At least two images are required to estimate depth maps")

        cv2 = self._load_cv2()
        focal_length = float(self.K[0, 0]) if self.K[0, 0] > 0 else 1.0
        depth_maps: List[np.ndarray] = []

        for index in range(len(images) - 1):
            left_image = images[index]
            right_image = images[index + 1]

            left_gray = self._to_grayscale(cv2, left_image)
            right_gray = self._to_grayscale(cv2, right_image)

            if left_gray.shape != right_gray.shape:
                right_gray = cv2.resize(right_gray, (left_gray.shape[1], left_gray.shape[0]))

            stereo = cv2.StereoSGBM_create(
                minDisparity=0,
                numDisparities=self.num_disparities,
                blockSize=self.block_size,
                P1=8 * self.block_size * self.block_size,
                P2=32 * self.block_size * self.block_size,
                disp12MaxDiff=1,
                uniquenessRatio=10,
                speckleWindowSize=100,
                speckleRange=32,
            )

            disparity = stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0
            depth_map = np.full(disparity.shape, np.nan, dtype=np.float32)
            valid_disparity = disparity > 0.0
            depth_map[valid_disparity] = (focal_length * self.baseline) / disparity[valid_disparity]
            depth_maps.append(depth_map)

        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            for index, depth_map in enumerate(depth_maps):
                np.save(self.output_dir / f"depth_map_{index:03d}.npy", depth_map)

        return depth_maps

    def run_sfm(self, matches: List[VerifiedMatches]) -> SparseCloud:
        if not matches:
            return SparseCloud(points_3d=np.empty((0, 3), dtype=np.float32), camera_poses={})

        points_3d = np.empty((0, 3), dtype=np.float32)
        camera_poses: Dict[str, np.ndarray] = {}
        return SparseCloud(points_3d=points_3d, camera_poses=camera_poses)