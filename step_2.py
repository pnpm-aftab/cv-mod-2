"""
Step 2: Object Measurement (Perspective Projection)
Module 2 Assignment - CSc 8830

Assignment step covered:
2) Implement a script to find real-world 2D dimensions of an object using
   perspective projection equations and camera calibration parameters.

What you need:
- camera_calibration.pkl from Step 1
- A test image containing a flat, rectangular object
- Known distance from camera to object plane (Z)

How to run:
Use via step_3.py (step_2 is imported as a module).

Outputs:
- results_step2/measurement_result.jpg (annotated image with dimensions)
"""

import numpy as np
import cv2
import pickle
import math
import os


class ObjectMeasurement:
    def __init__(self):
        """Initialize object measurement with camera calibration data."""
        calibration_file = "camera_calibration.pkl"
        with open(calibration_file, 'rb') as f:
            calib_data = pickle.load(f)
        
        self.mtx = calib_data["mtx"]  # Camera matrix
        self.dist = calib_data["dist"]  # Distortion coefficients
        self.square_size = calib_data["square_size"]

        print(f"Calibration: {calibration_file} (fx={self.mtx[0,0]:.0f}, fy={self.mtx[1,1]:.0f})")

    def _log(self, msg: str) -> None:
        print(msg)
    
    def undistort_image(self, img):
        """
        Undistort an image using camera calibration parameters
        
        Args:
            img: Input image
            
        Returns:
            Undistorted image
        """
        return cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
    
    def detect_object_corners(self, img):
        """
        Manually select 4 corners of a rectangular object in the image.
        Click in order: Top-Left, Top-Right, Bottom-Right, Bottom-Left.

        Returns:
            Array of 4 corner points or None if cancelled.
        """
        print("\n--- SELECT 4 CORNERS ---")
        print("Click in order: 1.Top-Left  2.Top-Right  3.Bottom-Right  4.Bottom-Left")
        print("Press 'r' to reset, 'q' to quit.")

        points = []
        display_img = img.copy()
        if len(display_img.shape) == 2:
            display_img = cv2.cvtColor(display_img, cv2.COLOR_GRAY2BGR)

        h, w = display_img.shape[:2]
        max_disp = 1000
        scale = 1.0
        if h > max_disp or w > max_disp:
            scale = max_disp / max(h, w)
            display_img = cv2.resize(display_img, (int(w * scale), int(h * scale)))

        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
                orig_x, orig_y = int(x / scale), int(y / scale)
                points.append((orig_x, orig_y))
                cv2.circle(display_img, (x, y), 5, (0, 255, 0), -1)
                cv2.putText(display_img, str(len(points)), (x + 10, y + 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("Select 4 Corners", display_img)

        cv2.namedWindow("Select 4 Corners")
        cv2.setMouseCallback("Select 4 Corners", mouse_callback)

        while True:
            cv2.imshow("Select 4 Corners", display_img)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('r'):
                points = []
                display_img = img.copy()
                if len(display_img.shape) == 2:
                    display_img = cv2.cvtColor(display_img, cv2.COLOR_GRAY2BGR)
                if scale != 1.0:
                    display_img = cv2.resize(display_img, (int(w * scale), int(h * scale)))
            if len(points) == 4:
                cv2.waitKey(500)
                break

        cv2.destroyAllWindows()
        if len(points) == 4:
            return np.array(points, dtype=np.float32)
        return None
    
    def calculate_real_world_dimensions(
        self,
        corners,
        known_distance,
        unit_label="units"
    ):
        """
        Calculate real-world dimensions using perspective projection
        
        The perspective projection equation relates image coordinates to real-world coordinates:
        
        For a point (X, Y, Z) in 3D world space, the image coordinates (u, v) are:
        u = fx * (X/Z) + cx
        v = fy * (Y/Z) + cy
        
        Where:
        - (fx, fy) are focal lengths in pixels
        - (cx, cy) is the principal point
        - Z is the distance from camera to the object
        
        Args:
            corners: Array of 4 corner points [TL, TR, BR, BL] in image coordinates
            known_distance: Distance from camera to object plane (Z) in same units as desired output
            
        Returns:
            Dictionary containing real-world width, height, and area
        """
        if corners is None or len(corners) != 4:
            print("ERROR: Invalid corners.")
            return None
        
        if known_distance is None:
            print("ERROR: known_distance is required for perspective projection.")
            return None

        # Get camera intrinsic parameters
        fx = self.mtx[0, 0]
        fy = self.mtx[1, 1]
        
        # Calculate dimensions in pixels (top edge for width, left edge for height)
        width_pixels = np.linalg.norm(corners[1] - corners[0])
        height_pixels = np.linalg.norm(corners[3] - corners[0])
        
        print(f"  Pixels: {width_pixels:.0f} x {height_pixels:.0f}")
        
        # Calculate real-world dimensions using perspective projection
        # From the pinhole camera model:
        # real_width = (pixel_width * Z) / focal_length
        
        width_real = (width_pixels * known_distance) / fx
        height_real = (height_pixels * known_distance) / fy
        
        # Calculate area
        area_real = width_real * height_real
        
        print(f"  Real: {width_real:.3f} x {height_real:.3f} {unit_label} (area {area_real:.4f})")
        
        return {
            'width_pixels': width_pixels,
            'height_pixels': height_pixels,
            'width_real': width_real,
            'height_real': height_real,
            'area_real': area_real,
            'distance': known_distance
        }
    
    def measure_object_in_image(
        self,
        image_path,
        known_distance,
        unit_label="units",
        save_result=False,
        output_dir="results_step2",
    ):
        """
        Complete workflow to measure an object in an image
        
        Args:
            image_path: Path to the test image
            known_distance: Distance from camera to object plane
            save_result: Whether to save the result image
            
        Returns:
            Dictionary containing measurement results
        """
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            print(f"ERROR: Could not load image from {image_path}")
            return None
        
        self._log(f"Image: {image_path} ({img.shape[1]}x{img.shape[0]})")
        self._log("Detecting object corners...")
        corners = self.detect_object_corners(img)
        
        if corners is None:
            print("Failed to detect object corners!")
            return None
        
        self._log("Calculating real-world dimensions...")
        results = self.calculate_real_world_dimensions(
            corners,
            known_distance,
            unit_label
        )
        
        # Save result image
        if save_result and results is not None:
            img_result = img.copy()
            
            # Draw the detected corners
            cv2.polylines(img_result, [corners.astype(int)], True, (0, 255, 0), 3)
            
            # Add dimension text
            text_width = f"Width: {results['width_real']:.2f} {unit_label}"
            text_height = f"Height: {results['height_real']:.2f} {unit_label}"
            if results.get("distance") is None:
                text_dist = "Distance: n/a"
            else:
                text_dist = f"Distance: {results['distance']:.2f} {unit_label}"
            
            cv2.putText(img_result, text_width, (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(img_result, text_height, (20, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(img_result, text_dist, (20, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, "measurement_result.jpg")
            cv2.imwrite(output_path, img_result)
            self._log(f"Result saved: {output_path}")

        return results
