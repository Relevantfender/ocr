#!/usr/bin/env python3
"""
Direct comparison: Draw both EasyOCR and PaddleOCR detections on the same image
with different colors to see alignment differences.
"""

import os
import cv2
import numpy as np
from main import extract_color, TARGET_NUMBER_COLOR, COLOR_TOLERANCE
from easyocr import Reader
from paddleocr import PaddleOCR

def draw_comparison(image_path):
    """Draw both EasyOCR and PaddleOCR bboxes on same image for comparison"""

    # Load image
    img = cv2.imread(image_path)
    h, w = img.shape[:2]

    # Extract and preprocess
    preprocessed = extract_color(img, TARGET_NUMBER_COLOR, COLOR_TOLERANCE)

    # Initialize models
    print("Loading models...")
    easy_reader = Reader(['en'], gpu=False)
    paddle_ocr = PaddleOCR(use_textline_orientation=False, lang='en')

    # Process with EasyOCR (normal polarity)
    print("\nProcessing with EasyOCR...")
    easy_results = easy_reader.readtext(preprocessed)
    easy_detections = []
    for bbox, text, conf in easy_results:
        text = text.strip()
        if text.isdigit():
            num = int(text)
            if 0 <= num <= 10:
                easy_detections.append((num, bbox, conf))

    print(f"EasyOCR found {len(easy_detections)} numbers")

    # Process with PaddleOCR (normal polarity)
    print("\nProcessing with PaddleOCR...")
    preprocessed_bgr = cv2.cvtColor(preprocessed, cv2.COLOR_GRAY2BGR)

    # CRITICAL FIX: Save to temp file and pass path instead of array
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        temp_path = tmp.name
        cv2.imwrite(temp_path, preprocessed_bgr)

    try:
        paddle_results = paddle_ocr.predict(temp_path)
    finally:
        os.unlink(temp_path)

    paddle_detections = []
    if paddle_results and isinstance(paddle_results, list) and len(paddle_results) > 0:
        result_obj = paddle_results[0]
        if 'rec_texts' in result_obj:
            rec_texts = result_obj['rec_texts']
            rec_polys = result_obj['rec_polys']
            rec_scores = result_obj['rec_scores']

            for text, bbox, score in zip(rec_texts, rec_polys, rec_scores):
                text = text.strip()
                if text.isdigit():
                    num = int(text)
                    if 0 <= num <= 10:
                        bbox_list = bbox.tolist() if hasattr(bbox, 'tolist') else bbox
                        paddle_detections.append((num, bbox_list, score))

    print(f"PaddleOCR found {len(paddle_detections)} numbers")

    # Create output canvas: preprocessed + original with overlaid boxes
    canvas = np.ones((h, w * 2, 3), dtype=np.uint8) * 255

    # Left: preprocessed grayscale to BGR
    preprocessed_display = cv2.cvtColor(preprocessed, cv2.COLOR_GRAY2BGR)
    canvas[:h, :w] = preprocessed_display

    # Right: original
    canvas[:h, w:] = img

    # Draw EasyOCR boxes in GREEN
    print("\n=== EasyOCR Detections (GREEN) ===")
    for num, bbox, conf in easy_detections:
        pts = np.array(bbox, dtype=np.int32)
        center_x = sum(pt[0] for pt in pts) / 4
        center_y = sum(pt[1] for pt in pts) / 4
        print(f"  Number '{num}' at ({center_x:.0f}, {center_y:.0f}) - confidence: {conf:.2f}")

        # Draw on preprocessed (left side)
        cv2.polylines(canvas, [pts], True, (0, 255, 0), 2)  # GREEN
        cv2.putText(canvas, f"E:{num}", (int(pts[0][0]), int(pts[0][1]) - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Draw on original (right side, with offset)
        pts_offset = pts.copy()
        pts_offset[:, 0] += w
        cv2.polylines(canvas, [pts_offset], True, (0, 255, 0), 2)  # GREEN
        cv2.putText(canvas, f"E:{num}", (int(pts_offset[0][0]), int(pts_offset[0][1]) - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Draw PaddleOCR boxes in RED
    print("\n=== PaddleOCR Detections (RED) ===")
    for num, bbox, conf in paddle_detections:
        pts = np.array(bbox, dtype=np.int32)
        center_x = sum(pt[0] for pt in bbox) / 4
        center_y = sum(pt[1] for pt in bbox) / 4
        print(f"  Number '{num}' at ({center_x:.0f}, {center_y:.0f}) - confidence: {conf:.2f}")

        # Draw on preprocessed (left side)
        cv2.polylines(canvas, [pts], True, (0, 0, 255), 3)  # RED (thicker)
        cv2.putText(canvas, f"P:{num}", (int(pts[0][0]) + 30, int(pts[0][1]) - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Draw on original (right side, with offset)
        pts_offset = pts.copy()
        pts_offset[:, 0] += w
        cv2.polylines(canvas, [pts_offset], True, (0, 0, 255), 3)  # RED (thicker)
        cv2.putText(canvas, f"P:{num}", (int(pts_offset[0][0]) + 30, int(pts_offset[0][1]) - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Add labels
    cv2.putText(canvas, "Preprocessed (Green=EasyOCR, Red=PaddleOCR)", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(canvas, "Original (Green=EasyOCR, Red=PaddleOCR)", (w + 10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    # Save
    output_path = image_path.replace('input', 'output/comparison')
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, canvas)

    print(f"\n✓ Comparison saved to: {output_path}")
    print("\nLegend:")
    print("  GREEN boxes (E:N) = EasyOCR detections")
    print("  RED boxes (P:N) = PaddleOCR detections")
    print("\nIf RED boxes are offset from the actual numbers, PaddleOCR has alignment issues.")
    print("If GREEN and RED boxes overlap on same numbers, both models agree.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python compare_models.py <image_path>")
        print("Example: python compare_models.py input/Cipelici.png")
    else:
        draw_comparison(sys.argv[1])
