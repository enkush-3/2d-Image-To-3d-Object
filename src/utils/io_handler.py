
from __future__ import annotations

import importlib
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np


DEFAULT_IMAGE_EXTENSIONS: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


class IOHandler:
    def __init__(
        self,
        input_dir: str = "data/inputs",
        output_dir: str = "data/outputs",
        image_extensions: Sequence[str] = DEFAULT_IMAGE_EXTENSIONS,
    ) -> None:
        project_root = Path(__file__).resolve().parents[2]
        self.input_dir = (project_root / input_dir).resolve()
        self.output_dir = (project_root / output_dir).resolve()
        self.image_extensions = tuple(ext.lower() for ext in image_extensions)

    def list_image_files(self) -> List[Path]:
        if not self.input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {self.input_dir}")

        files = [
            path
            for path in sorted(self.input_dir.iterdir())
            if path.is_file() and path.suffix.lower() in self.image_extensions
        ]
        if not files:
            raise FileNotFoundError(
                f"No images found in {self.input_dir}. Supported formats: {', '.join(self.image_extensions)}"
            )
        return files

    def load_images(self, max_images: Optional[int] = None) -> Tuple[List[np.ndarray], List[str]]:
        try:
            cv2 = importlib.import_module("cv2")
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "OpenCV (cv2) is required to load images. Install it with: pip install opencv-python"
            ) from exc

        image_files = self.list_image_files()
        if max_images is not None and max_images > 0:
            image_files = image_files[:max_images]

        images: List[np.ndarray] = []
        image_names: List[str] = []

        for image_path in image_files:
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Failed to load image: {image_path}")
            images.append(image)
            image_names.append(image_path.name)

        return images, image_names

    def ensure_output_dir(self) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        return self.output_dir

    def save_sparse_cloud(self, points_3d: np.ndarray, camera_poses: dict, file_name: str = "sparse_cloud.npz") -> Path:
        output_dir = self.ensure_output_dir()
        output_path = output_dir / file_name
        np.savez_compressed(output_path, points_3d=points_3d, camera_poses=camera_poses)
        return output_path

