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

os.environ.setdefault("OPENCV_OPENCL_RUNTIME", "disabled")  # avoid noisy/buggy OpenCL paths on macOS
import cv2
import numpy as np


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


def _find_chessboard(gray: np.ndarray, pattern_size: tuple[int, int]):
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray, pattern_size, flags=flags)
    return found, corners


def main() -> int:
    # Work around occasional macOS OpenCL cache write failures in opencv-python
    try:
        if hasattr(cv2, "ocl") and hasattr(cv2.ocl, "setUseOpenCL"):
            cv2.ocl.setUseOpenCL(False)
    except Exception:
        pass

    # Configuration
    IMAGES_DIR = "calibration_images"
    PATTERN_SIZE = (10, 6)  # inner corners (cols, rows) = 11x7 squares
    SQUARE_SIZE_M = 0.015
    OUTPUT_FILE = "camera_calibration.pkl"

    image_paths = _list_images(IMAGES_DIR)
    print(f"Found {len(image_paths)} images in '{IMAGES_DIR}'")
    if not image_paths:
        print("ERROR: No images found. Put images into the folder and try again.")
        return 1

    pattern_size = PATTERN_SIZE
    square_size = float(SQUARE_SIZE_M)
    if square_size <= 0:
        print("ERROR: square_size must be > 0")
        return 2

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)

    # Prepare object points (0..cols-1, 0..rows-1) scaled by square size
    # builds coordinates of the chessboard corners in the object space
    cols, rows = pattern_size
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= square_size

    obj_points: list[np.ndarray] = []
    img_points: list[np.ndarray] = []
    image_size = None

    for idx, fname in enumerate(image_paths, 1):
        img = cv2.imread(fname)
        if img is None:
            continue
        base = os.path.basename(fname)
        # convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if image_size is None:
            image_size = gray.shape[::-1]

        # find chessboard corners
        found, corners = _find_chessboard(gray, pattern_size)

        if found and corners is not None:
            # refine corners on the grayscale image.
            try:
                corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            except Exception:
                pass
            obj_points.append(objp.copy())
            img_points.append(corners)

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
            cv2.imshow("Chessboard detections", vis)
            cv2.waitKey(200)

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

