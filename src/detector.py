"""
Number Detector Module

Detects numbers 0-10 in images using OCR and returns their bounding box locations.
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict
import easyocr


class NumberDetector:
    """
    Detects single and double-digit numbers (0-10) in images.

    Attributes:
        reader: EasyOCR reader instance
        allowed_numbers: Set of valid numbers to detect (0-10)
    """

    def __init__(self, languages=['en']):
        """
        Initialize the number detector.

        Args:
            languages: List of languages for OCR (default: ['en'])
        """
        print("Initializing EasyOCR reader...")
        self.reader = easyocr.Reader(languages, gpu=False)
        self.allowed_numbers = set(range(11))  # 0 through 10
        print("Number detector initialized!")

    def detect(self, image_path: str, confidence_threshold: float = 0.5) -> List[Dict]:
        """
        Detect numbers 0-10 in an image.

        Args:
            image_path: Path to the input image
            confidence_threshold: Minimum confidence score (0-1)

        Returns:
            List of detections, each containing:
                - number: The detected number
                - bbox: Bounding box coordinates [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                - confidence: Detection confidence score
                - center: Center point of the bounding box (x, y)
        """
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")

        # Perform OCR
        results = self.reader.readtext(image)

        # Filter for valid numbers
        detections = []
        for bbox, text, confidence in results:
            # Clean the text and try to parse as integer
            cleaned_text = text.strip()

            # Skip if confidence is too low
            if confidence < confidence_threshold:
                continue

            try:
                number = int(cleaned_text)

                # Only keep numbers 0-10
                if number in self.allowed_numbers:
                    # Calculate center point
                    bbox_array = np.array(bbox)
                    center = tuple(bbox_array.mean(axis=0).astype(int))

                    detection = {
                        'number': number,
                        'bbox': bbox,
                        'confidence': confidence,
                        'center': center
                    }
                    detections.append(detection)

            except ValueError:
                # Not a valid integer, skip
                continue

        return detections

    def detect_from_array(self, image: np.ndarray, confidence_threshold: float = 0.5) -> List[Dict]:
        """
        Detect numbers 0-10 in a numpy array image.

        Args:
            image: Input image as numpy array
            confidence_threshold: Minimum confidence score (0-1)

        Returns:
            List of detections (same format as detect method)
        """
        # Perform OCR
        results = self.reader.readtext(image)

        # Filter for valid numbers
        detections = []
        for bbox, text, confidence in results:
            cleaned_text = text.strip()

            if confidence < confidence_threshold:
                continue

            try:
                number = int(cleaned_text)

                if number in self.allowed_numbers:
                    bbox_array = np.array(bbox)
                    center = tuple(bbox_array.mean(axis=0).astype(int))

                    detection = {
                        'number': number,
                        'bbox': bbox,
                        'confidence': confidence,
                        'center': center
                    }
                    detections.append(detection)

            except ValueError:
                continue

        return detections

    def visualize_detections(self, image_path: str, detections: List[Dict],
                            output_path: str = None) -> np.ndarray:
        """
        Visualize detected numbers on the image.

        Args:
            image_path: Path to the input image
            detections: List of detections from detect()
            output_path: Optional path to save the visualization

        Returns:
            Image with bounding boxes and labels drawn
        """
        image = cv2.imread(image_path)

        for det in detections:
            bbox = np.array(det['bbox'], dtype=np.int32)
            number = det['number']
            confidence = det['confidence']
            center = det['center']

            # Draw bounding box
            cv2.polylines(image, [bbox], True, (0, 255, 0), 2)

            # Draw label
            label = f"{number} ({confidence:.2f})"
            cv2.putText(image, label, (bbox[0][0], bbox[0][1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Draw center point
            cv2.circle(image, center, 5, (255, 0, 0), -1)

        if output_path:
            cv2.imwrite(output_path, image)

        return image


# Example usage
if __name__ == "__main__":
    # Test the detector
    detector = NumberDetector()

    print("\nNumber Detector is ready!")
    print("Allowed numbers: 0-10")
    print("\nUsage:")
    print("  detections = detector.detect('image.jpg')")
    print("  detector.visualize_detections('image.jpg', detections, 'output.jpg')")
