"""
Step 3: Validation Experiment
Module 2 Assignment - CSc 8830

Assignment step covered:
3) Validate using an experiment where you image an object from a known
   distance (> 2 meters) and compare measured dimensions to ground truth.

What you need:
- camera_calibration.pkl from Step 1
- A flat rectangular object with known dimensions (ground truth)
- A measured camera distance > 2 meters
- One or more test images at that distance
- image inside the ../test_images_step_3 folder

How to run (defaults shown):
python step-3.py

Outputs:
- results_step3/measurement_result.jpg (annotated image)
- results_step3/validation_report.json (detailed results)
- results_step3/validation_summary.txt (human-readable summary)
"""

import numpy as np
import cv2
import pickle
import json
import os
from datetime import datetime
from step_2 import ObjectMeasurement


class ValidationExperiment:
    def __init__(self, calibration_file="camera_calibration.pkl", verbose: bool = False):
        """
        Initialize validation experiment
        
        Args:
            calibration_file: Path to camera calibration data
        """
        self.measurer = ObjectMeasurement(calibration_file, verbose=verbose)
        self.results = []
        self.verbose = verbose

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)
        
    def create_test_object_info(self, width_real, height_real, units='meters'):
        """
        Create ground truth information for test object
        
        Args:
            width_real: Actual width of the test object
            height_real: Actual height of the test object
            units: Unit of measurement (meters, centimeters, etc.)
            
        Returns:
            Dictionary with ground truth data
        """
        return {
            'width_real': width_real,
            'height_real': height_real,
            'area_real': width_real * height_real,
            'units': units
        }
    
    def run_single_validation(self, image_path, distance, ground_truth, 
                              visualize=False, save_result=True):
        """
        Run a single validation experiment
        
        Args:
            image_path: Path to test image
            distance: Distance from camera to object (must be > 2 meters)
            ground_truth: Dictionary with actual object dimensions
            visualize: Show detection results
            save_result: Save result image
            
        Returns:
            Dictionary with validation results
        """
        if distance < 2.0:
            print("WARNING: Distance is less than 2 meters. Assignment requires > 2 meters.")
        if self.verbose:
            print("\nVALIDATION EXPERIMENT")
            print(f"Test Image: {image_path}")
            print(f"Distance: {distance:.2f} {ground_truth['units']}")
            print(f"Ground Truth Width: {ground_truth['width_real']:.2f} {ground_truth['units']}")
            print(f"Ground Truth Height: {ground_truth['height_real']:.2f} {ground_truth['units']}")
        
        # Measure the object
        measurement = self.measurer.measure_object_in_image(
            image_path=image_path,
            known_distance=distance,
            visualize=visualize,
            save_result=save_result,
            output_dir="results_step3",
        )
        
        if measurement is None:
            print("Measurement failed.")
            return None
        
        # Calculate errors
        width_error = abs(measurement['width_real'] - ground_truth['width_real'])
        height_error = abs(measurement['height_real'] - ground_truth['height_real'])
        area_error = abs(measurement['area_real'] - ground_truth['area_real'])
        
        width_error_pct = (width_error / ground_truth['width_real']) * 100
        height_error_pct = (height_error / ground_truth['height_real']) * 100
        area_error_pct = (area_error / ground_truth['area_real']) * 100
        
        # Compile results
        result = {
            'image_path': image_path,
            'distance': distance,
            'ground_truth': ground_truth,
            'measured': measurement,
            'errors': {
                'width_abs': width_error,
                'width_pct': width_error_pct,
                'height_abs': height_error,
                'height_pct': height_error_pct,
                'area_abs': area_error,
                'area_pct': area_error_pct
            },
            'timestamp': datetime.now().isoformat()
        }
        
        self.print_validation_results(result)
        
        return result
    
    def print_validation_results(self, result):
        """
        Print validation results in a formatted way
        
        Args:
            result: Validation result dictionary
        """
        print("Validation results:")
        print(
            f"- Ground truth (W x H): "
            f"{result['ground_truth']['width_real']:.2f} x {result['ground_truth']['height_real']:.2f} "
            f"{result['ground_truth']['units']}"
        )
        print(
            f"- Measured (W x H): "
            f"{result['measured']['width_real']:.2f} x {result['measured']['height_real']:.2f} "
            f"{result['ground_truth']['units']}"
        )
        print(
            f"- Errors (%): W {result['errors']['width_pct']:.2f}%, "
            f"H {result['errors']['height_pct']:.2f}%, "
            f"A {result['errors']['area_pct']:.2f}%"
        )
        
        # Determine if validation passed
        width_passed = result['errors']['width_pct'] < 5.0  # Less than 5% error
        height_passed = result['errors']['height_pct'] < 5.0
        
        print(
            f"Validation status: {'PASS' if (width_passed and height_passed) else 'FAIL'} "
            "(< 5% error on width and height)"
        )
    
    def save_validation_report(self, results, output_file="results_step3/validation_report.json"):
        """
        Save validation results to JSON file
        
        Args:
            results: List of validation result dictionaries
            output_file: Output file path
        """
        # Convert numpy types to Python types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            return obj
        
        results_converted = convert_types(results)

        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(results_converted, f, indent=2)
        
        self._log(f"Validation report saved to: {output_file}")
    
    def create_summary_report(self, results, output_file="results_step3/validation_summary.txt"):
        """
        Create a human-readable summary report
        
        Args:
            results: List of validation result dictionaries
            output_file: Output file path
        """
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(output_file, "w") as f:
            f.write("="*70 + "\n")
            f.write("VALIDATION EXPERIMENT SUMMARY REPORT\n")
            f.write("Module 2 Assignment - CSc 8830: Computer Vision\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Number of tests: {len(results)}\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for i, result in enumerate(results, 1):
                f.write("-"*70 + "\n")
                f.write(f"TEST {i}: {os.path.basename(result['image_path'])}\n")
                f.write("-"*70 + "\n")
                f.write(f"Distance: {result['distance']:.2f} {result['ground_truth']['units']}\n\n")
                
                f.write("Ground Truth:\n")
                f.write(f"  Width: {result['ground_truth']['width_real']:.2f} {result['ground_truth']['units']}\n")
                f.write(f"  Height: {result['ground_truth']['height_real']:.2f} {result['ground_truth']['units']}\n")
                f.write(f"  Area: {result['ground_truth']['area_real']:.2f} sq {result['ground_truth']['units']}\n\n")
                
                f.write("Measured:\n")
                f.write(f"  Width: {result['measured']['width_real']:.2f} {result['ground_truth']['units']}\n")
                f.write(f"  Height: {result['measured']['height_real']:.2f} {result['ground_truth']['units']}\n")
                f.write(f"  Area: {result['measured']['area_real']:.2f} sq {result['ground_truth']['units']}\n\n")
                
                f.write("Errors:\n")
                f.write(f"  Width: {result['errors']['width_abs']:.2f} {result['ground_truth']['units']} "
                       f"({result['errors']['width_pct']:.2f}%)\n")
                f.write(f"  Height: {result['errors']['height_abs']:.2f} {result['ground_truth']['units']} "
                       f"({result['errors']['height_pct']:.2f}%)\n")
                f.write(f"  Area: {result['errors']['area_abs']:.2f} sq {result['ground_truth']['units']} "
                       f"({result['errors']['area_pct']:.2f}%)\n\n")
                
                width_passed = result['errors']['width_pct'] < 5.0
                height_passed = result['errors']['height_pct'] < 5.0
                f.write(f"Status: {'PASS' if (width_passed and height_passed) else 'FAIL'}\n")
                f.write("\n")
            
            # Overall statistics
            f.write("="*70 + "\n")
            f.write("OVERALL STATISTICS\n")
            f.write("="*70 + "\n")
            
            if results:
                avg_width_error = np.mean([r['errors']['width_pct'] for r in results])
                avg_height_error = np.mean([r['errors']['height_pct'] for r in results])
                avg_area_error = np.mean([r['errors']['area_pct'] for r in results])
                
                passed_count = sum(1 for r in results 
                                  if r['errors']['width_pct'] < 5.0 and r['errors']['height_pct'] < 5.0)
                
                f.write(f"Average Width Error: {avg_width_error:.2f}%\n")
                f.write(f"Average Height Error: {avg_height_error:.2f}%\n")
                f.write(f"Average Area Error: {avg_area_error:.2f}%\n")
                f.write(f"Tests Passed: {passed_count}/{len(results)}\n")
                f.write(f"Success Rate: {(passed_count/len(results)*100):.1f}%\n")
        
        self._log(f"Summary report saved to: {output_file}")


def main():
    """
    Main validation experiment function
    """
    print("STEP 3: VALIDATION EXPERIMENT")
    
    # Initialize validation experiment
    try:
        validator = ValidationExperiment("camera_calibration.pkl", verbose=False)
    except FileNotFoundError:
        print("ERROR: camera_calibration.pkl not found.")
        print("Run Step 1 (camera calibration) first.")
        return
    except Exception as e:
        print(f"ERROR: Failed to initialize: {e}")
        return
    
    # Configuration for validation experiment
    # IMPORTANT: Measure these values accurately for your test object!
    
    # Ground truth object: 13in x 10in
    # Convert to meters: 0.3302m x 0.254m
    TEST_OBJECT_WIDTH = 0.3302  # meters
    TEST_OBJECT_HEIGHT = 0.254  # meters
    TEST_OBJECT_UNITS = "meters"
    
    # Distance from camera to object (must be > 2 meters as per assignment)
    # Measure this accurately using a tape measure or laser measurer
    CAMERA_DISTANCE = 2.35  # meters (> 2.0 as required)
    
    # Test image paths (automatically tests all images in the folder)
    TEST_IMAGES_DIR = 'test_images_step_3'
    TEST_IMAGE_PATHS = [
        os.path.join(TEST_IMAGES_DIR, f) 
        for f in os.listdir(TEST_IMAGES_DIR) 
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]
    TEST_IMAGE_PATHS.sort()
    
    # Optional: Reference object for better accuracy
    # Example: Credit card width = 85.6mm = 0.0856m
    REFERENCE_WIDTH_PIXELS = None  # Measure in image if using
    REFERENCE_WIDTH_REAL = None    # Real width in meters
    
    # Create ground truth
    ground_truth = validator.create_test_object_info(
        width_real=TEST_OBJECT_WIDTH,
        height_real=TEST_OBJECT_HEIGHT,
        units=TEST_OBJECT_UNITS
    )
    
    print(
        f"Validation config: object {TEST_OBJECT_WIDTH} x {TEST_OBJECT_HEIGHT} {TEST_OBJECT_UNITS}, "
        f"distance {CAMERA_DISTANCE} {TEST_OBJECT_UNITS}"
    )
    
    # Check if test images exist
    missing_images = [p for p in TEST_IMAGE_PATHS if not os.path.exists(p)]
    if missing_images:
        print("ERROR: Test image(s) not found:")
        for p in missing_images:
            print(f"- {p}")
        print("Fix: capture images at the configured distance and update TEST_IMAGE_PATHS.")
        return
    
    # Run the validation experiment for each image
    for image_path in TEST_IMAGE_PATHS:
        result = validator.run_single_validation(
            image_path=image_path,
            distance=CAMERA_DISTANCE,
            ground_truth=ground_truth,
            visualize=False,  # Set to True to see corner detection
            save_result=True
        )
        
        if result:
            validator.results.append(result)
        else:
            print("Validation experiment failed for one image.")
            print("Troubleshooting: check lighting, framing, distance, calibration.")
    
    if validator.results:
        # Save results
        validator.save_validation_report(validator.results)
        validator.create_summary_report(validator.results)
        
        print("Validation experiment completed.")
        print(
            "Generated files: results_step3/measurement_result.jpg, "
            "results_step3/validation_report.json, results_step3/validation_summary.txt"
        )
    else:
        print("Validation experiment failed for all images.")
        print("Troubleshooting: check lighting, framing, distance, calibration.")


if __name__ == "__main__":
    main()
