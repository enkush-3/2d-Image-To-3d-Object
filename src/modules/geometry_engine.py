from __future__ import annotations

import importlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from utils.config import SparseCloud, VerifiedMatches


class GeometryEngine:
    """
    Stage 4 — Хөдөлгөөнөөс бүтэц  (run_sfm)
    Stage 5 — Гүний зураглал       (estimate_depth_maps)
    """

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
        self.K = calibration_matrix.astype(np.float64)
        self.baseline = float(baseline)
        self.num_disparities = self._normalize_num_disparities(num_disparities)
        self.block_size = self._normalize_block_size(block_size)
        self.output_dir = Path(output_dir) if output_dir else None

    # ── helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _normalize_num_disparities(n: int) -> int:
        v = max(16, int(n))
        return v + (16 - v % 16) % 16

    @staticmethod
    def _normalize_block_size(b: int) -> int:
        v = max(3, int(b))
        return v if v % 2 == 1 else v + 1

    @staticmethod
    def _to_grayscale(cv2, img: np.ndarray) -> np.ndarray:
        return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def _load_cv2(self):
        try:
            return importlib.import_module("cv2")
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("pip install opencv-python") from exc

    @staticmethod
    def _filter_points(pts: np.ndarray, z_min=0.01, z_max=1000.0) -> np.ndarray:
        fin = np.isfinite(pts).all(axis=1)
        z   = (pts[:, 2] > z_min) & (pts[:, 2] < z_max)
        return pts[fin & z]

    @staticmethod
    def _masked_points(p1, p2, mask):
        m = mask.ravel().astype(bool)
        return p1[m], p2[m]

    @staticmethod
    def _decide_pnp_order(ia, ib, pa, pb, known):
        if ia in known and ib not in known:
            return ia, ib, pa, pb
        if ib in known and ia not in known:
            return ib, ia, pb, pa
        return None, None, None, None

    # ── Stage 5 — Depth Map Estimation ────────────────────────────────

    def estimate_depth_maps(self, images: List[np.ndarray]) -> List[np.ndarray]:
        if len(images) < 2:
            raise ValueError("Хамгийн багадаа 2 зураг шаардлагатай.")
        cv2 = self._load_cv2()
        focal = float(self.K[0, 0]) if self.K[0, 0] > 0 else 1.0
        depth_maps: List[np.ndarray] = []

        print(f"\n[Depth Estimation] Эхэлж байна... {len(images)-1} хос.")
        for idx in range(len(images) - 1):
            left  = self._to_grayscale(cv2, images[idx])
            right = self._to_grayscale(cv2, images[idx + 1])
            if left.shape != right.shape:
                right = cv2.resize(right, (left.shape[1], left.shape[0]))

            stereo = cv2.StereoSGBM_create(
                minDisparity=0,
                numDisparities=self.num_disparities,
                blockSize=self.block_size,
                P1=8  * self.block_size ** 2,
                P2=32 * self.block_size ** 2,
                disp12MaxDiff=1,
                uniquenessRatio=10,
                speckleWindowSize=100,
                speckleRange=32,
            )
            disp = stereo.compute(left, right).astype(np.float32) / 16.0
            dm = np.full(disp.shape, np.nan, dtype=np.float32)
            valid = disp > 0.0
            dm[valid] = (focal * self.baseline) / disp[valid]
            depth_maps.append(dm)
            print(f"  Хос {idx:03d}-{idx+1:03d}: хүчинтэй {valid.sum()} пиксел "
                  f"({100*valid.sum()/dm.size:.1f}%)")

        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            for i, dm in enumerate(depth_maps):
                p = self.output_dir / f"depth_map_{i:03d}.npy"
                np.save(p, dm)
                print(f"  Хадгалагдлаа: {p}")

        print(f"[Depth Estimation] Дууслаа. {len(depth_maps)} depth map.\n")
        return depth_maps

    # ── Stage 4 — Structure from Motion ───────────────────────────────

    def run_sfm(self, matches: List[VerifiedMatches]) -> SparseCloud:
        """
        VerifiedMatches.inlier_points() ашиглан incremental SfM хийнэ.
        Feature_Engine нь keypoints1/keypoints2-г автоматаар дамжуулдаг.
        """
        if not matches:
            print("[SfM] Тохирол байхгүй.")
            return SparseCloud(points_3d=np.empty((0,3), dtype=np.float32), camera_poses={})

        cv2 = self._load_cv2()
        K = self.K
        print(f"\n[SfM] Эхэлж байна... {len(matches)} хос.")

        poses: Dict[int, np.ndarray] = {}
        cloud: List[np.ndarray] = []

        # ── анхны хос ─────────────────────────────────────────────────
        first = matches[0]
        i0, i1 = first.image_pair
        pts0, pts1 = first.inlier_points()          # ← config.py-н метод

        E, e_mask = cv2.findEssentialMat(
            pts0, pts1, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        if E is None:
            print("[SfM] Essential matrix олдсонгүй.")
            return SparseCloud(points_3d=np.empty((0,3), dtype=np.float32), camera_poses={})

        _, R, t, p_mask = cv2.recoverPose(E, pts0, pts1, K, mask=e_mask)
        P0 = np.hstack([np.eye(3), np.zeros((3,1))])
        P1 = np.hstack([R, t])
        poses[i0], poses[i1] = P0, P1

        m = p_mask.ravel().astype(bool)
        if m.sum() >= 4:
            p4d = cv2.triangulatePoints(K@P0, K@P1, pts0[m].T, pts1[m].T)
            p3d = self._filter_points((p4d[:3]/p4d[3]).T.astype(np.float32))
            cloud.append(p3d)
            print(f"  Анхны хос ({i0},{i1}): {len(p3d)} 3D цэг.")

        # ── нэмэгдэл хос ──────────────────────────────────────────────
        for match in matches[1:]:
            ia, ib = match.image_pair
            pa, pb = match.inlier_points()

            ref, new, p_ref, p_new = self._decide_pnp_order(ia, ib, pa, pb, poses)
            if ref is None:
                print(f"  Хос ({ia},{ib}): хоёулаа шинэ — алгаслаа.")
                continue

            P_ref = poses[ref]
            ok, P_new = self._pnp_pose(cv2, K, P_ref, p_ref, p_new, cloud)
            if not ok:
                P_new = self._triangulate_pose(P_ref, p_ref, p_new, K, cv2)
            if P_new is None:
                print(f"  Хос ({ia},{ib}): байршил тодорхойлогдсонгүй.")
                continue

            poses[new] = P_new
            if len(p_ref) >= 4:
                p4d = cv2.triangulatePoints(K@P_ref, K@P_new, p_ref.T, p_new.T)
                p3d = self._filter_points((p4d[:3]/p4d[3]).T.astype(np.float32))
                cloud.append(p3d)
                print(f"  Хос ({ia},{ib}): {len(p3d)} шинэ цэг, камер {new}.")

        # ── нэгтгэх ───────────────────────────────────────────────────
        pts_out = (np.vstack(cloud).astype(np.float32)
                   if cloud else np.empty((0,3), dtype=np.float32))
        poses_out = {f"camera_{k:04d}": v for k,v in poses.items()}

        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            np.save(self.output_dir / "sparse_points_3d.npy", pts_out)

        print(f"[SfM] Дууслаа. 3D цэг: {len(pts_out)}, Камер: {len(poses_out)}\n")
        return SparseCloud(points_3d=pts_out, camera_poses=poses_out)

    # ── PnP / triangulation туслахууд ─────────────────────────────────

    def _pnp_pose(self, cv2, K, P_ref, pts_ref, pts_new, cloud):
        combined = np.vstack(cloud)
        n = min(len(pts_new), len(combined))
        if n < 4:
            return False, None
        try:
            ok, rvec, tvec, inliers = cv2.solvePnPRansac(
                combined[:n].astype(np.float64),
                pts_new[:n].astype(np.float64),
                K, None,
                confidence=0.99, reprojectionError=8.0, iterationsCount=200,
            )
        except cv2.error:
            return False, None
        if not ok or inliers is None:
            return False, None
        R, _ = cv2.Rodrigues(rvec)
        return True, np.hstack([R, tvec])

    def _triangulate_pose(self, P_ref, pts_ref, pts_new, K, cv2):
        if len(pts_ref) < 8:
            return None
        E, _ = cv2.findEssentialMat(
            pts_ref, pts_new, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        if E is None:
            return None
        _, R, t, _ = cv2.recoverPose(E, pts_ref, pts_new, K)
        return np.hstack([R, t])


# ── Backward-compatible wrapper ────────────────────────────────────────

def run_sfm_with_features(geometry_engine, matches, features=None):
    """keypoints1/2 matches дотор байгаа тул features параметр шаардлагагүй."""
    return geometry_engine.run_sfm(matches)