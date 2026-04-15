from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
import cv2 as cv

@dataclass
class FeatureData:
    """Container for extracted features from a single image."""
    keypoints: List[cv.KeyPoint]
    descriptors: Optional[np.ndarray]
    image_index: int
    image_name: str

@dataclass
class Geometry_Engine:
    """Container for verified matches between an image pair."""
    matches: List[cv.DMatch]
    inliers: List[cv.DMatch]
    mask: np.ndarray
    image_pair: Tuple[int, int]

@dataclass
class Surface_Engine:
    """Sparse 3D point cloud and camera poses (placeholder for SfM)."""
    points_3d: Optional[np.ndarray] = None
    cameras: Optional[List] = None
    point_colors: Optional[np.ndarray] = None