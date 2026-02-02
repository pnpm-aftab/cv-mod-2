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

How to run (defaults shown):
python step-2.py

Outputs:
- results_step2/measurement_result.jpg (annotated image with dimensions)
"""

import numpy as np
import cv2
import pickle
import math
import os


class ObjectMeasurement:
    def __init__(self, calibration_file="camera_calibration.pkl", verbose: bool = False):
        """
        Initialize object measurement with camera calibration data
        
        Args:
            calibration_file: Path to the calibration data file
        """
        # Load calibration data
        with open(calibration_file, 'rb') as f:
            calib_data = pickle.load(f)
        
        self.mtx = calib_data["mtx"]  # Camera matrix
        self.dist = calib_data["dist"]  # Distortion coefficients
        self.square_size = calib_data["square_size"]
        self.verbose = verbose

        if self.verbose:
            print(f"Camera calibration loaded from {calibration_file}")
            print(f"Focal length: fx={self.mtx[0,0]:.2f}, fy={self.mtx[1,1]:.2f}")
            print(f"Principal point: cx={self.mtx[0,2]:.2f}, cy={self.mtx[1,2]:.2f}")

    def _log(self, msg: str) -> None:
        if self.verbose:
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
    
    def detect_object_corners(self, img, visualize=False):
        """
        Detect corners of a rectangular object in the image
        
        Args:
            img: Input image (grayscale or color)
            visualize: Whether to show the detection result
            
        Returns:
            List of 4 corner points in clockwise order starting from top-left
            or None if detection fails
        """
        # Convert to grayscale if needed
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
        
        # Manual selection if automatic fails
        def manual_selection(image, original_img):
            print("\n--- MANUAL SELECTION MODE ---")
            print("Click the 4 corners of the card in this order:")
            print("1. Top-Left  2. Top-Right  3. Bottom-Right  4. Bottom-Left")
            print("Press 'r' to reset or 'q' to quit.")
            
            points = []
            display_img = original_img.copy()
            if len(display_img.shape) == 2:
                display_img = cv2.cvtColor(display_img, cv2.COLOR_GRAY2BGR)
            
            # Scale down for display if image is too large
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
                    display_img = original_img.copy()
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

        # Try automatic detection first
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply Canny edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) > 0:
            # Find the largest contour (assuming it's our object)
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Approximate the contour to a polygon
            epsilon = 0.02 * cv2.arcLength(largest_contour, True)
            approx = cv2.approxPolyDP(largest_contour, epsilon, True)
            
            # If we have 4 corners, use them
            if len(approx) == 4:
                corners = approx.reshape(4, 2).astype(np.float32)
                return self._order_corners(corners)
            
            self._log(f"Automatic detection found {len(approx)} corners instead of 4.")
        else:
            self._log("No contours detected automatically.")

        # Fallback to manual selection
        # Note: We pass the original image (img) to manual_selection
        return manual_selection(img, img)
    
    def _order_corners(self, corners):
        """
        Order corners in clockwise direction starting from top-left
        
        Args:
            corners: Array of 4 corner points
            
        Returns:
            Ordered corners [TL, TR, BR, BL]
        """
        # Sort by x coordinate to separate left and right
        corners = corners[np.argsort(corners[:, 0])]
        
        # Left points (indices 0 and 1) and right points (indices 2 and 3)
        left_corners = corners[:2]
        right_corners = corners[2:]
        
        # Sort left corners by y (top then bottom)
        left_corners = left_corners[np.argsort(left_corners[:, 1])]
        
        # Sort right corners by y (top then bottom)
        right_corners = right_corners[np.argsort(right_corners[:, 1])]
        
        # Return in order: TL, TR, BR, BL
        return np.array([left_corners[0], right_corners[0], right_corners[1], left_corners[1]])
    
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
        
        # Calculate dimensions in pixels
        # Width (using top edge between TL and TR)
        width_pixels = np.linalg.norm(corners[1] - corners[0])
        
        # Height (using left edge between TL and BL)
        height_pixels = np.linalg.norm(corners[3] - corners[0])
        
        # Also calculate using bottom and right edges for verification
        width_pixels_bottom = np.linalg.norm(corners[2] - corners[3])
        height_pixels_right = np.linalg.norm(corners[2] - corners[1])
        
        # Average the measurements
        width_pixels = (width_pixels + width_pixels_bottom) / 2
        height_pixels = (height_pixels + height_pixels_right) / 2
        
        if self.verbose:
            print("\nPixel measurements:")
            print(f"Width: {width_pixels:.2f} pixels")
            print(f"Height: {height_pixels:.2f} pixels")
        
        # Calculate real-world dimensions using perspective projection
        # From the pinhole camera model:
        # real_width = (pixel_width * Z) / focal_length
        
        width_real = (width_pixels * known_distance) / fx
        height_real = (height_pixels * known_distance) / fy
        
        if self.verbose:
            print(f"\nUsing perspective projection (Z = {known_distance:.2f} {unit_label}):")
            print("Formula: real_size = (pixel_size * Z) / focal_length")
        
        # Calculate area
        area_real = width_real * height_real
        
        if self.verbose:
            print("\nReal-world dimensions:")
            print(f"Width: {width_real:.2f} {unit_label}")
            print(f"Height: {height_real:.2f} {unit_label}")
            print(f"Area: {area_real:.2f} square {unit_label}")
        
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
        visualize=False,
        save_result=False,
        output_dir="results_step2",
    ):
        """
        Complete workflow to measure an object in an image
        
        Args:
            image_path: Path to the test image
            known_distance: Distance from camera to object plane
            visualize: Whether to show intermediate results
            save_result: Whether to save the result image
            
        Returns:
            Dictionary containing measurement results
        """
        self._log("=" * 60)
        self._log("OBJECT MEASUREMENT - STEP 2")
        self._log("=" * 60)
        
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            print(f"ERROR: Could not load image from {image_path}")
            return None
        
        self._log(f"\nImage loaded: {image_path}")
        self._log(f"Image size: {img.shape[1]}x{img.shape[0]}")
        
        # Detect object corners directly on the original image
        self._log("\nDetecting object corners...")
        corners = self.detect_object_corners(img, visualize=visualize)
        
        if corners is None:
            print("Failed to detect object corners!")
            return None
        
        # Calculate real-world dimensions
        self._log("\nCalculating real-world dimensions...")
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
            self._log(f"\nResult image saved to: {output_path}")
        
        self._log("=" * 60)
        
        return results


def main():
    """
    Main function to demonstrate object measurement
    """
    print("STEP 2: OBJECT MEASUREMENT (Perspective Projection)")
    
    # Configuration
    CALIBRATION_FILE = "camera_calibration.pkl"
    TEST_IMAGE = "test_images_step_2/validation_test.jpeg"

    # Use inches for this run (keep all units consistent)
    UNIT_LABEL = "in"

    # Distance from camera to object plane (measured in inches)
    # For Step 2, provide the distance you measured during your experiment
    KNOWN_DISTANCE = 12.0  # Example: 12 inches

    # Card size (in inches): 2.0 x 3.5 (Width x Height)
    CARD_WIDTH_REAL = 2.0
    CARD_HEIGHT_REAL = 3.5
    
    # Create object measurement instance
    try:
        measurer = ObjectMeasurement(CALIBRATION_FILE, verbose=False)
    except FileNotFoundError:
        print(f"ERROR: Calibration file '{CALIBRATION_FILE}' not found.")
        print("Run Step 1 (camera calibration) first.")
        return
    except Exception as e:
        print(f"ERROR: Failed to load calibration data: {e}")
        return
    
    # Measure the object
    results = measurer.measure_object_in_image(
        image_path=TEST_IMAGE,
        known_distance=KNOWN_DISTANCE,
        unit_label=UNIT_LABEL,
        visualize=False,  # Set to True to see corner detection
        save_result=True
    )
    
    if results:
        eval_w, eval_h = results['width_real'], results['height_real']
        gt_w, gt_h = CARD_WIDTH_REAL, CARD_HEIGHT_REAL

        print("Measurement completed successfully.")
        print(f"Distance:     {KNOWN_DISTANCE:.2f} {UNIT_LABEL}")
        print(f"Ground Truth: {gt_w:.2f} x {gt_h:.2f} {UNIT_LABEL}")
        print(f"Evaluated:    {eval_w:.2f} x {eval_h:.2f} {UNIT_LABEL}")
        
        width_error = abs(eval_w - gt_w) / gt_w * 100
        height_error = abs(eval_h - gt_h) / gt_h * 100
        print(f"Error:        {width_error:.2f}% width, {height_error:.2f}% height")
    else:
        print("Measurement failed.")
        print("Troubleshooting: check lighting, object visibility, and distance settings.")


if __name__ == "__main__":
    main()
