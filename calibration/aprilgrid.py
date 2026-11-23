"""Detect and fit the tabletop 6×6 AprilTag calibration target."""

from itertools import product
import json
from pathlib import Path

import cv2
import numpy as np


WORLD_DEFINITION = "Tag 0 centre; +X toward decreasing rows, +Y toward decreasing columns, +Z out of the board"
REFERENCE_SIZE = 590


class AprilGrid:
    def __init__(self, path: Path, reference: Path | None = None):
        self.geometry = json.loads(path.read_text())
        size = self.geometry["tag_size_m"]
        pitch = self.geometry["tag_pitch_m"]
        if not np.isfinite([size, pitch]).all() or not 0 < size < pitch:
            raise ValueError("AprilGrid requires 0 < tag_size_m < tag_pitch_m")
        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        self.centres = np.array(list(product(range(6), repeat=2)), dtype=float) * pitch
        # The printed tags all have the same orientation. OpenCV starts at the
        # physical lower-left corner, then follows the left edge upwards.
        offsets = size / 2 * np.array([[-1, 1], [-1, -1], [1, -1], [1, 1]])
        self.planar_corners = self.centres[:, None, :] + offsets
        shape = self.planar_corners.shape[:2]
        self.world_corners = np.dstack(
            (-self.planar_corners[..., 1], -self.planar_corners[..., 0], np.zeros(shape))
        )
        margin = pitch / 2 + .0025
        low, high = -margin, 5 * pitch + margin
        self.reference_corners = np.array([[low, low], [high, low], [high, high], [low, high]], np.float32)
        self.reference_pixels = REFERENCE_SIZE * np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32)
        self.grid_to_reference = cv2.getPerspectiveTransform(self.reference_corners, self.reference_pixels)
        self.features = cv2.SIFT_create(nfeatures=3000, contrastThreshold=.012)
        self.reference_features = None
        if reference is not None:
            image = cv2.imread(str(reference), cv2.IMREAD_GRAYSCALE)
            if image is None or image.shape != (REFERENCE_SIZE, REFERENCE_SIZE):
                raise ValueError(f"Invalid AprilGrid reference image: {reference}")
            self.reference_features = self.features.detectAndCompute(image, None)
            if self.reference_features[1] is None:
                raise ValueError(f"No AprilGrid features in reference image: {reference}")

    def make_reference(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        seed = self.seed_homography(gray)
        if seed is None:
            raise ValueError("Cannot find AprilGrid in the reference frame")
        corners = cv2.perspectiveTransform(self.reference_corners[None], seed)[0]
        warp = cv2.getPerspectiveTransform(corners, self.reference_pixels)
        return cv2.warpPerspective(gray, warp, (REFERENCE_SIZE, REFERENCE_SIZE))

    def locate_reference(self, gray):
        reference_points, reference_descriptors = self.reference_features
        points, descriptors = self.features.detectAndCompute(gray, None)
        if descriptors is None or len(descriptors) < 2:
            return None
        pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(reference_descriptors, descriptors, k=2)
        matches = [first for first, second in pairs if first.distance < .77 * second.distance]
        if len(matches) < 12:
            return None
        source = np.array([reference_points[match.queryIdx].pt for match in matches])
        target = np.array([points[match.trainIdx].pt for match in matches])
        homography, inliers = cv2.findHomography(source, target, cv2.USAC_MAGSAC, 2, maxIters=4000)
        if homography is None or inliers.sum() < 12:
            return None
        return homography @ self.grid_to_reference

    def seed_homography(self, gray):
        found = {}
        for scale in (1, 2, 3):
            enlarged = gray if scale == 1 else cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            for window in (5, 9, 13, 19, 25, 31, 41):
                aruco = cv2.aruco
                params = aruco.DetectorParameters() if hasattr(aruco, "ArucoDetector") else aruco.DetectorParameters_create()
                params.adaptiveThreshWinSizeMin = params.adaptiveThreshWinSizeMax = window
                params.minMarkerPerimeterRate = .012
                if hasattr(aruco, "ArucoDetector"):
                    corners, ids, _ = aruco.ArucoDetector(self.dictionary, params).detectMarkers(enlarged)
                else:
                    corners, ids, _ = aruco.detectMarkers(enlarged, self.dictionary, parameters=params)
                if ids is None:
                    continue
                for marker_id, quad in zip(ids.flat, corners):
                    if 0 <= marker_id < len(self.centres):
                        found.setdefault(int(marker_id), quad.reshape(4, 2) / scale)
                if len(found) >= 6:
                    break
            if len(found) >= 6:
                break
        if not found:
            return None
        # Marker centres give correspondences near the true grid positions
        # even when the adaptive threshold clips the printed black border.
        ids = list(found)
        centres = [
            cv2.perspectiveTransform(
                np.array([[[.5, .5]]], dtype=np.float32),
                cv2.getPerspectiveTransform(np.array([[0, 1], [0, 0], [1, 0], [1, 1]], np.float32), found[index]),
            )[0, 0]
            for index in ids
        ]
        if len(found) >= 4:
            homography, _ = cv2.findHomography(self.centres[ids], np.array(centres), cv2.RANSAC, 2)
            if homography is not None:
                return homography
        return cv2.getPerspectiveTransform(self.planar_corners[ids[0]].astype(np.float32), found[ids[0]])

    def correspondences(self, gray):
        seed = self.locate_reference(gray) if self.reference_features is not None else self.seed_homography(gray)
        if seed is None:
            return None
        expected = cv2.perspectiveTransform(self.planar_corners.reshape(1, -1, 2), seed).reshape(-1, 4, 2)
        matched_ids, matched_corners = [], []
        height, width = gray.shape
        for index, predicted in enumerate(expected):
            quad = predicted.astype(np.float32)
            edge = np.linalg.norm(quad - np.roll(quad, 1, axis=0), axis=1).min()
            window = int(np.clip(round(.13 * edge), 3, 12))
            if edge < 16 or np.any(quad < window + 1) or np.any(quad >= np.array([width, height]) - (window + 1)):
                continue
            refined = cv2.cornerSubPix(
                gray, quad.copy(), (window, window), (-1, -1),
                (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 35, .01),
            )
            if np.linalg.norm(refined - quad, axis=1).max() > window:
                continue
            # The printed registration squares touch each tag diagonally.
            # Only use corners supported by this black/white saddle in the image.
            u = .10 * (np.roll(refined, 1, axis=0) - refined)
            v = .10 * (np.roll(refined, -1, axis=0) - refined)
            samples = np.stack((refined + u + v, refined - u - v, refined + u - v, refined - u + v))
            intensities = cv2.remap(gray, samples[..., 0], samples[..., 1], cv2.INTER_LINEAR).astype(float)
            contrast = intensities[2:].min(axis=0) - intensities[:2].max(axis=0)
            if np.min(contrast) < 25:
                continue
            matched_ids.append(index)
            matched_corners.append(refined)
        if len(matched_ids) < 2:
            return None
        return np.array(matched_ids), self.world_corners[matched_ids], np.array(matched_corners)

    def estimate_pose(self, image, intrinsics):
        if image.shape[:2] != (intrinsics["height"], intrinsics["width"]):
            raise ValueError("Image dimensions do not match recorded intrinsics")
        matches = self.correspondences(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        if matches is None:
            return None
        ids, world, pixels = matches
        matrix = np.array([
            [intrinsics["fx"], 0, intrinsics["cx"]],
            [0, intrinsics["fy"], intrinsics["cy"]],
            [0, 0, 1.],
        ])
        distortion = np.asarray(intrinsics["coeffs"], dtype=float)
        world = world.reshape(-1, 3)
        pixels = pixels.reshape(-1, 2)
        # The target is planar. Initialize with the planar solver and discard
        # entire tags if any of their corners disagree with the fit.
        ok, rotation, translation = cv2.solvePnP(world, pixels, matrix, distortion, flags=cv2.SOLVEPNP_IPPE)
        if not ok or translation[2, 0] <= 0:
            return None
        projected, _ = cv2.projectPoints(world, rotation, translation, matrix, distortion)
        errors = np.linalg.norm(projected.reshape(-1, 2) - pixels, axis=1)
        good_tags = (errors.reshape(-1, 4) < 1.5).all(axis=1)
        if good_tags.sum() < 2:
            return None
        good_corners = np.repeat(good_tags, 4)
        world, pixels = world[good_corners], pixels[good_corners]
        rotation, translation = cv2.solvePnPRefineLM(world, pixels, matrix, distortion, rotation, translation)
        projected, _ = cv2.projectPoints(world, rotation, translation, matrix, distortion)
        rmse = float(np.sqrt(np.mean(np.sum((projected.reshape(-1, 2) - pixels) ** 2, axis=1))))
        world_T_camera = np.eye(4)
        world_T_camera[:3, :3] = cv2.Rodrigues(rotation)[0].T
        world_T_camera[:3, 3] = (-world_T_camera[:3, :3] @ translation).ravel()
        if world_T_camera[2, 3] <= 0:
            return None
        return world_T_camera, rmse, ids[good_tags].tolist()
