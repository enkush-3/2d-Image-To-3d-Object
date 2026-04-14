from __future__ import annotations

from typing import List, Optional, Tuple
import cv2 as cv
import numpy as np

from utils.config import FeatureData, VerifiedMatches


class Feature_Engine:

    def __init__(
        self,
        akaze_threshold: float = 0.001,
        akaze_n_octaves: int = 4,
        akaze_n_octave_layers: int = 4,
        akaze_descriptor_type: int = cv.AKAZE_DESCRIPTOR_MLDB,
        ratio_test_threshold: float = 0.75,
        match_pairs: Optional[List[Tuple[int, int]]] = None,
        verbose: bool = True,
    ):
        self.akaze = cv.AKAZE_create(
            threshold=akaze_threshold,
            nOctaves=akaze_n_octaves,
            nOctaveLayers=akaze_n_octave_layers,
            descriptor_type=akaze_descriptor_type,
        )
        self.ratio_test_threshold = ratio_test_threshold
        self.match_pairs = match_pairs
        self.verbose = verbose

    def extract_features(self, images: List[np.ndarray]) -> List[FeatureData]:
        if self.verbose:
            print(f"\n[AKAZE Feature Extraction] Эхэлж байна... Нийт {len(images)} зураг.")
        
        features: List[FeatureData] = []
        for idx, img in enumerate(images):
            if len(img.shape) == 3:
                gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            else:
                gray = img

            kp, desc = self.akaze.detectAndCompute(gray, None)

            if self.verbose:
                num_kp = len(kp) if kp is not None else 0
                desc_shape = desc.shape if desc is not None else (0, 0)
                print(f"  Зураг {idx:04d}: {num_kp:4d} түлхүүр цэг, дескриптор хэмжээ {desc_shape}")

            features.append(
                FeatureData(
                    keypoints=kp,
                    descriptors=desc,
                    image_index=idx,
                    image_name=f"img_{idx:04d}",
                )
            )
        
        if self.verbose:
            print(f"[AKAZE Feature Extraction] Дууслаа. {len(features)} зурагны онцлог танигдсан.\n")
        return features

    def match_features(self, features: List[FeatureData]) -> List[VerifiedMatches]:
        if self.match_pairs is None:
            num_images = len(features)
            pairs = [(i, i + 1) for i in range(num_images - 1)]
        else:
            pairs = self.match_pairs

        if self.verbose:
            print(f"[Feature Matching] Эхэлж байна... Нийт {len(pairs)} хос зураг.")
            print(f"  Хослолууд: {pairs}")

        matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)

        verified_matches_list: List[VerifiedMatches] = []

        for i, j in pairs:
            if self.verbose:
                print(f"\n  Хос ({i}, {j}) боловсруулж байна...")

            desc1 = features[i].descriptors
            desc2 = features[j].descriptors

            if desc1 is None or desc2 is None:
                if self.verbose:
                    print(f"    !!! Дескриптор байхгүй тул алгасаж байна.")
                continue

            raw_matches = matcher.knnMatch(desc1, desc2, k=2)
            if self.verbose:
                print(f"    KNN тохируулга: {len(raw_matches)} ширхэг")

            good_matches = []
            for m, n in raw_matches:
                if m.distance < self.ratio_test_threshold * n.distance:
                    good_matches.append(m)

            if self.verbose:
                print(f"    Харьцаа шалгалтаар үлдсэн: {len(good_matches)}")

            if len(good_matches) < 8:
                if self.verbose:
                    print(f"    !!! Хангалттай тохирол байхгүй (<8). Алгаслаа.")
                continue

            kp1 = features[i].keypoints
            kp2 = features[j].keypoints

            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)

            F, inlier_mask = cv.findFundamentalMat(
                src_pts, dst_pts, cv.FM_RANSAC, ransacReprojThreshold=3.0, confidence=0.99
            )

            if F is None or inlier_mask is None:
                if self.verbose:
                    print(f"    !!! Фундаментал матриц олдсонгүй.")
                continue

            inlier_mask = inlier_mask.ravel().astype(bool)
            inliers = [m for m, is_inlier in zip(good_matches, inlier_mask) if is_inlier]

            if self.verbose:
                print(f"    Геометр баталгаажуулалтаар үлдсэн inliers: {len(inliers)}")

            if len(inliers) < 8:
                if self.verbose:
                    print(f"    !!! Inlier тоо хангалтгүй (<8). Алгаслаа.")
                continue

            verified_matches_list.append(
                VerifiedMatches(
                    matches=good_matches,
                    inliers=inliers,
                    mask=inlier_mask,
                    image_pair=(i, j),
                )
            )
            if self.verbose:
                print(f"    ✓ Амжилттай хос: ({i}, {j}) - {len(inliers)} inlier-тэй.")

        if self.verbose:
            print(f"\n[Feature Matching] Дууслаа. {len(verified_matches_list)} хос амжилттай таарч, баталгаажсан.\n")
        return verified_matches_list