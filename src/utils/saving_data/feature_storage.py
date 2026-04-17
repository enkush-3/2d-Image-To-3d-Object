# utils/feature_storage.py
from __future__ import annotations

import os
import numpy as np
import cv2 as cv
from typing import List, Tuple

from utils.config import FeatureData, VerifiedMatches


class FeatureStorage:
    """AKAZE онцлог ба тохируулгын өгөгдлийг хадгалах, унших класс."""

    @staticmethod
    def keypoints_to_array(keypoints: List[cv.KeyPoint]) -> np.ndarray:
        """cv.KeyPoint жагсаалтыг (N, 7) хэмжээтэй numpy массив болгон хөрвүүлнэ."""
        if not keypoints:
            return np.empty((0, 7), dtype=np.float32)
        arr = np.array([(kp.pt[0], kp.pt[1], kp.size, kp.angle,
                         kp.response, kp.octave, kp.class_id)
                        for kp in keypoints], dtype=np.float32)
        return arr

    @staticmethod
    def array_to_keypoints(arr: np.ndarray) -> List[cv.KeyPoint]:
        """(N, 7) массиваас cv.KeyPoint жагсаалт үүсгэнэ."""
        if arr.size == 0:
            return []
        keypoints = []
        for row in arr:
            kp = cv.KeyPoint(x=row[0], y=row[1], size=row[2], angle=row[3],
                             response=row[4], octave=int(row[5]), class_id=int(row[6]))
            keypoints.append(kp)
        return keypoints

    @staticmethod
    def save(
        features: List[FeatureData],
        matches: List[VerifiedMatches],
        output_dir: str,
        base_name: str = "feature_data"
    ) -> str:
        """
        Онцлог цэг, дескриптор, тохируулсан хосын мэдээллийг файлд хадгална.

        Args:
            features: extract_features()-ийн буцаасан FeatureData жагсаалт.
            matches: match_features()-ийн буцаасан VerifiedMatches жагсаалт.
            output_dir: Хадгалах хавтас.
            base_name: Файлын үндсэн нэр (өргөтгөлгүй).

        Returns:
            Хадгалсан файлын бүтэн зам.
        """
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f"{base_name}.npz")

        # Онцлог бүрийн өгөгдлийг цуглуулна
        kp_arrays = []          # keypoints массивуудын жагсаалт
        desc_list = []          # descriptors массивууд (None байж болно)
        img_indices = []        # зурагны индекс
        img_names = []          # зурагны нэр

        for feat in features:
            kp_arr = FeatureStorage.keypoints_to_array(feat.keypoints)
            kp_arrays.append(kp_arr)
            desc = feat.descriptors if feat.descriptors is not None else np.empty((0, 0), dtype=np.uint8)
            desc_list.append(desc)
            img_indices.append(feat.image_index)
            img_names.append(feat.image_name)

        # Матч мэдээллийг цуглуулна
        match_pairs = []        # хос (i, j)
        all_matches_data = []   # good_matches-ийн массивууд
        all_inliers_data = []   # inliers-ийн массивууд

        for vm in matches:
            i, j = vm.image_pair
            match_pairs.append((i, j))

            # good_matches -> (N, 3) массив
            good_arr = np.array([(m.queryIdx, m.trainIdx, m.distance) for m in vm.matches], dtype=np.float32)
            all_matches_data.append(good_arr)

            # inliers -> (M, 3) массив
            inl_arr = np.array([(m.queryIdx, m.trainIdx, m.distance) for m in vm.inliers], dtype=np.float32)
            all_inliers_data.append(inl_arr)

        # NPZ файлд хадгалах
        np.savez_compressed(
            save_path,
            # Features хэсэг
            num_images=len(features),
            img_indices=np.array(img_indices, dtype=np.int32),
            img_names=np.array(img_names, dtype=object),
            keypoints=np.array(kp_arrays, dtype=object),
            descriptors=np.array(desc_list, dtype=object),
            # Matches хэсэг
            num_pairs=len(matches),
            match_pairs=np.array(match_pairs, dtype=np.int32),
            matches_data=np.array(all_matches_data, dtype=object),
            inliers_data=np.array(all_inliers_data, dtype=object),
        )
        print(f"[FeatureStorage] Өгөгдөл '{save_path}' файлд хадгалагдлаа.")
        return save_path

    @staticmethod
    def load(load_path: str) -> Tuple[List[FeatureData], List[VerifiedMatches]]:
        """
        save() ашиглан хадгалсан өгөгдлийг ачаалж,
        FeatureData болон VerifiedMatches жагсаалт буцаана.
        """
        data = np.load(load_path, allow_pickle=True)

        # Features сэргээх
        num_images = data['num_images'].item()
        img_indices = data['img_indices'].tolist()
        img_names = data['img_names'].tolist()
        kp_arrays = data['keypoints']
        desc_arrays = data['descriptors']

        features: List[FeatureData] = []
        for i in range(num_images):
            kp = FeatureStorage.array_to_keypoints(kp_arrays[i])
            desc = desc_arrays[i] if desc_arrays[i].size > 0 else None
            features.append(FeatureData(
                keypoints=kp,
                descriptors=desc,
                image_index=img_indices[i],
                image_name=img_names[i]
            ))

        # Matches сэргээх
        num_pairs = data['num_pairs'].item()
        match_pairs = data['match_pairs'].tolist()
        matches_data = data['matches_data']
        inliers_data = data['inliers_data']

        verified_matches: List[VerifiedMatches] = []
        for p in range(num_pairs):
            i, j = match_pairs[p]
            # good_matches
            good_arr = matches_data[p]
            good_matches = [cv.DMatch(int(q), int(t), d) for q, t, d in good_arr]
            # inliers
            inl_arr = inliers_data[p]
            inliers = [cv.DMatch(int(q), int(t), d) for q, t, d in inl_arr]
            # mask үүсгэх (good_matches-ийн урттай)
            mask = np.zeros(len(good_matches), dtype=bool)
            inlier_qidx = {m.queryIdx for m in inliers}
            for idx, m in enumerate(good_matches):
                if m.queryIdx in inlier_qidx:
                    mask[idx] = True

            verified_matches.append(VerifiedMatches(
                matches=good_matches,
                inliers=inliers,
                mask=mask,
                image_pair=(i, j)
            ))

        print(f"[FeatureStorage] '{load_path}' файлаас {num_images} зурагны онцлог, {num_pairs} хос тохируулга ачаалагдлаа.")
        return features, verified_matches