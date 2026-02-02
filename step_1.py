"""
Step 1: Camera Calibration (Chessboard)
Module 2 Assignment - CSc 8830

Assignment step covered:
1) Perform camera calibration using OpenCV with a smartphone camera.

What you need:
- images inside the ../calibration_images folder
- The chessboard square size (edge length) is set to 0.015 m (15 mm).

How to run:
python step_1.py

Outputs:
- camera_calibration.pkl (camera intrinsics + distortion)
"""

from __future__ import annotations

import glob
import os
import pickle
from dataclasses import dataclass
from typing import Iterable

os.environ.setdefault("OPENCV_OPENCL_RUNTIME", "disabled")  # avoid noisy/buggy OpenCL paths on macOS
import cv2  # noqa: E402
import numpy as np


@dataclass(frozen=True)
class PatternResult:
    pattern_size: tuple[int, int]  # (cols, rows) inner corners
    successes: int


def _list_images(images_dir: str) -> list[str]:
    patterns = [
        os.path.join(images_dir, "*.jpg"),
        os.path.join(images_dir, "*.jpeg"),
        os.path.join(images_dir, "*.png"),
    ]
    files: list[str] = []
    for p in patterns:
        files.extend(glob.glob(p))
    return sorted(set(files))


def _candidate_patterns() -> list[tuple[int, int]]:
    # (cols, rows) INNER corners. Includes common OpenCV and calib.io variants.
    return [
        (10, 6),  # common for "7x11 squares" style
        (9, 6),
        (8, 6),
        (7, 6),
        (6, 9),
        (6, 10),
        (11, 7),
        (7, 11),
        (5, 8),
        (4, 11),
    ]


def _resize_gray(gray: np.ndarray, max_dim: int) -> tuple[np.ndarray, float]:
    if max_dim <= 0:
        return gray, 1.0
    h, w = gray.shape[:2]
    m = max(h, w)
    if m <= max_dim:
        return gray, 1.0
    scale = max_dim / float(m)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA), scale


def _find_chessboard(gray: np.ndarray, pattern_size: tuple[int, int], *, fast: bool, max_dim: int):
    """
    fast=True is used for pattern auto-detection scoring (speed > accuracy).
    fast=False is used for the final detection pass.
    """
    gray_small, scale = _resize_gray(gray, max_dim=max_dim)

    if fast:
        # Fast path: classic detector with FAST_CHECK on downscaled image.
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK
        found, corners = cv2.findChessboardCorners(gray_small, pattern_size, flags=flags)
        return found, corners, scale

    # Final pass: prefer SB if available (more robust), but avoid EXHAUSTIVE (can be very slow).
    if hasattr(cv2, "findChessboardCornersSB"):
        flags = cv2.CALIB_CB_ACCURACY | cv2.CALIB_CB_NORMALIZE_IMAGE
        found, corners = cv2.findChessboardCornersSB(gray_small, pattern_size, flags=flags)
        return found, corners, scale

    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray_small, pattern_size, flags=flags)
    return found, corners, scale


def _score_patterns(
    image_paths: Iterable[str],
    patterns: list[tuple[int, int]],
    *,
    max_dim: int,
    verbose: bool,
) -> list[PatternResult]:
    results: list[PatternResult] = []
    for pat in patterns:
        if verbose:
            print(f"Scoring pattern {pat[0]}x{pat[1]} ...")
        ok = 0
        for fname in image_paths:
            img = cv2.imread(fname)
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            found, _, _ = _find_chessboard(gray, pat, fast=True, max_dim=max_dim)
            if found:
                ok += 1
        results.append(PatternResult(pattern_size=pat, successes=ok))
    results.sort(key=lambda r: r.successes, reverse=True)
    return results


def main() -> int:
    # Work around occasional macOS OpenCL cache write failures in opencv-python
    try:
        if hasattr(cv2, "ocl") and hasattr(cv2.ocl, "setUseOpenCL"):
            cv2.ocl.setUseOpenCL(False)
    except Exception:
        pass

    # Configuration (Defaults)
    IMAGES_DIR = "calibration_images"
    PATTERN = "auto"
    SQUARE_SIZE_M = 0.015
    MAX_DIM = 1200
    SHOW = False
    SAVE_DEBUG = False
    OUTPUT_FILE = "camera_calibration.pkl"
    VERBOSE = False

    image_paths = _list_images(IMAGES_DIR)
    print(f"Found {len(image_paths)} images in '{IMAGES_DIR}'")
    if not image_paths:
        print("ERROR: No images found. Put images into the folder and try again.")
        return 1

    if PATTERN.lower().strip() == "auto":
        scored = _score_patterns(image_paths, _candidate_patterns(), max_dim=int(MAX_DIM), verbose=VERBOSE)
        best = scored[0]
        if VERBOSE:
            print("Pattern auto-detect results (top 5):")
            for r in scored[:5]:
                print(f"  - {r.pattern_size[0]}x{r.pattern_size[1]}: {r.successes}/{len(image_paths)} detections")
        if best.successes == 0:
            print("\nERROR: No chessboard detections for common patterns.")
            print("Most likely the CHESSBOARD inner-corner size is different.")
            return 2
        pattern_size = best.pattern_size
        print(f"Using pattern (inner corners): {pattern_size[0]}x{pattern_size[1]}")
    else:
        try:
            cols, rows = PATTERN.lower().split("x")
            pattern_size = (int(cols), int(rows))
        except Exception:
            print('ERROR: PATTERN must be "auto" or like "10x6"')
            return 2

    square_size = float(SQUARE_SIZE_M)
    if square_size <= 0:
        print("ERROR: square_size must be > 0")
        return 2

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)

    # Prepare object points (0..cols-1, 0..rows-1) scaled by square size
    cols, rows = pattern_size
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= square_size

    obj_points: list[np.ndarray] = []
    img_points: list[np.ndarray] = []
    image_size = None

    debug_dir = os.path.join("results", "chessboard_detections")
    if SAVE_DEBUG:
        os.makedirs(debug_dir, exist_ok=True)

    for idx, fname in enumerate(image_paths, 1):
        img = cv2.imread(fname)
        if img is None:
            if VERBOSE:
                print(f"Skip unreadable: {fname}")
            continue
        base = os.path.basename(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if image_size is None:
            image_size = gray.shape[::-1]

        found, corners, scale = _find_chessboard(gray, pattern_size, fast=False, max_dim=int(MAX_DIM))

        if found and corners is not None:
            # If we detected on a downscaled image, map corner coordinates back to the original.
            if scale != 1.0:
                corners = corners.astype(np.float32) / float(scale)

            # Refine corners on the ORIGINAL-resolution grayscale image.
            try:
                corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            except Exception:
                pass
            obj_points.append(objp.copy())
            img_points.append(corners)
            if VERBOSE:
                print(f"Image {idx}/{len(image_paths)}: detected ({base})")

            if SHOW or SAVE_DEBUG:
                vis = img.copy()
                cv2.drawChessboardCorners(vis, pattern_size, corners, found)
                cv2.putText(
                    vis,
                    f"DETECTED {pattern_size[0]}x{pattern_size[1]}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                if SHOW:
                    cv2.imshow("Chessboard detections", vis)
                    cv2.waitKey(200)
                if SAVE_DEBUG:
                    out = os.path.join(debug_dir, f"{idx:03d}_DETECTED_{base}.jpg")
                    cv2.imwrite(out, vis)
        else:
            if VERBOSE:
                print(f"Image {idx}/{len(image_paths)}: NOT detected ({base})")
            if SAVE_DEBUG:
                vis = img.copy()
                cv2.putText(
                    vis,
                    f"NOT DETECTED {pattern_size[0]}x{pattern_size[1]}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
                out = os.path.join(debug_dir, f"{idx:03d}_NOT_DETECTED_{base}.jpg")
                cv2.imwrite(out, vis)

    if SHOW:
        cv2.destroyAllWindows()

    if not obj_points:
        print("ERROR: No valid detections; cannot calibrate.")
        return 3

    print("Calibrating camera...")
    if image_size is None:
        print("ERROR: Could not determine image size.")
        return 3
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, image_size, None, None)

    print("\n" + "=" * 60)
    print("CALIBRATION RESULTS (CHESSBOARD)")
    print("=" * 60)
    print(f"Detections used: {len(obj_points)}/{len(image_paths)}")
    print(f"Reprojection error: {ret:.6f}")
    print(f"\nCamera matrix:\n{mtx}")
    print(f"\nDistortion coefficients:\n{dist}")
    print("=" * 60)

    data = {
        "ret": ret,
        "mtx": mtx,
        "dist": dist,
        "rvecs": rvecs,
        "tvecs": tvecs,
        "square_size": square_size,
        "chessboard_size": pattern_size,  # inner corners
        "calibration_images_dir": IMAGES_DIR,
    }
    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(data, f)
    print(f"\nSaved calibration to: {OUTPUT_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

