"""
Test script for the Number Detector component.

This script demonstrates how to use the NumberDetector class.
"""

from src.detector import NumberDetector
import cv2
import numpy as np


def create_sample_image():
    """Create a simple test image with numbers 0-10."""
    # Create white background
    img = np.ones((400, 600, 3), dtype=np.uint8) * 255

    # Add some numbers
    font = cv2.FONT_HERSHEY_SIMPLEX
    numbers = [0, 3, 5, 7, 10]
    positions = [(50, 100), (200, 100), (350, 100), (150, 250), (400, 250)]

    for num, pos in zip(numbers, positions):
        cv2.putText(img, str(num), pos, font, 3, (0, 0, 0), 3)

    # Save test image
    cv2.imwrite('tests/test_images/sample_numbers.jpg', img)
    print("Created sample test image: tests/test_images/sample_numbers.jpg")
    return img


def test_detector():
    """Test the number detector on a sample image."""
    print("=" * 60)
    print("Testing Number Detector Component")
    print("=" * 60)

    # Create sample image
    create_sample_image()

    # Initialize detector
    detector = NumberDetector()

    # Detect numbers
    print("\n[1] Detecting numbers in image...")
    detections = detector.detect('tests/test_images/sample_numbers.jpg')

    print(f"\n[2] Found {len(detections)} numbers:")
    for i, det in enumerate(detections, 1):
        print(f"  {i}. Number: {det['number']}, "
              f"Confidence: {det['confidence']:.2f}, "
              f"Center: {det['center']}")

    # Visualize
    print("\n[3] Creating visualization...")
    detector.visualize_detections(
        'tests/test_images/sample_numbers.jpg',
        detections,
        'output/detections_visualized.jpg'
    )
    print("  Saved to: output/detections_visualized.jpg")

    print("\n" + "=" * 60)
    print("Component 1 (Number Detector) is working!")
    print("=" * 60)


if __name__ == "__main__":
    test_detector()
