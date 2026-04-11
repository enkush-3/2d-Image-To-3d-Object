from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

import numpy as np

from utils.config import FeatureData, VerifiedMatches, SparseCloud


class FeatureEngineInterface(ABC):
    @abstractmethod
    def extract_features(self, images: List[np.ndarray]) -> List[FeatureData]:
        """Extract keypoints and descriptors from input images."""

    @abstractmethod
    def match_features(self, features: List[FeatureData]) -> List[VerifiedMatches]:
        """Match descriptors between images and verify matches."""


class GeometryEngineInterface(ABC):
    @abstractmethod
    def estimate_depth_maps(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """Estimate depth maps from consecutive image pairs."""

    @abstractmethod
    def run_sfm(self, matches: List[VerifiedMatches]) -> SparseCloud:
        """Estimate sparse 3D points and camera poses using SfM."""


class SurfaceEngineInterface(ABC):
    @abstractmethod
    def generate_dense_mesh(self, sparse_data: SparseCloud, images: List[np.ndarray]) -> str:
        """Generate dense 3D mesh and return mesh file path."""

    @abstractmethod
    def visualize(self, mesh_path: str) -> None:
        """Visualize or preview generated mesh."""
