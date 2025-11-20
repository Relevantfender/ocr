"""
Simple OCR comparison tool for numbers 0-10
Compares EasyOCR, Tesseract, and PaddleOCR
"""

import cv2
import numpy as np
import os
from pathlib import Path
import easyocr
import pytesseract
from paddleocr import PaddleOCR

# Color mapping for numbers (will be used later for floodFill)
COLORS = {
    1: '#D94E4E',  # warm muted red
    2: '#3A77C2',  # deep denim blue
    3: '#E7C85A',  # soft golden yellow
    4: '#4FA36E',  # medium jade green
}

def hex_to_bgr(hex_color):
    """Convert hex color to BGR for OpenCV"""
    hex_color = hex_color.lstrip('#')
    rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return (rgb[2], rgb[1], rgb[0])  # Convert RGB to BGR

def get_images_from_folder(folder='input'):
    """Load all images from input folder"""
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    images = []

    if not os.path.exists(folder):
        return None

    for file in os.listdir(folder):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            images.append(os.path.join(folder, file))

    return images if images else None

def draw_bounding_boxes(image, detections, model_name):
    """
    Draw bounding boxes on image with color-coded numbers and legend

    Args:
        image: Original image
        detections: List of (number, bbox) tuples
        model_name: Name of the OCR model

    Returns:
        Image with bounding boxes and legend
    """
    img = image.copy()
    h, w = img.shape[:2]

    # Add space on the right for legend
    legend_width = 200
    canvas = np.ones((h, w + legend_width, 3), dtype=np.uint8) * 255
    canvas[:h, :w] = img

    # Draw bounding boxes
    for number, bbox in detections:
        # Get color based on number (cycle through 4 colors)
        color_key = ((number - 1) % 4) + 1
        color_hex = COLORS[color_key]
        color_bgr = hex_to_bgr(color_hex)

        # Draw bounding box
        if len(bbox) == 4 and len(bbox[0]) == 2:  # Polygon format
            pts = np.array(bbox, dtype=np.int32)
            cv2.polylines(canvas, [pts], True, color_bgr, 2)
        else:  # Rectangle format (x, y, w, h)
            x, y, bw, bh = bbox
            cv2.rectangle(canvas, (x, y), (x + bw, y + bh), color_bgr, 2)

        # Draw number label
        label_pos = bbox[0] if len(bbox) == 4 else (bbox[0], bbox[1])
        cv2.putText(canvas, str(number), (int(label_pos[0]), int(label_pos[1]) - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_bgr, 2)

    # Draw legend
    legend_x = w + 10
    cv2.putText(canvas, "Legend", (legend_x, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    y_offset = 60
    for region_num, hex_color in COLORS.items():
        color_bgr = hex_to_bgr(hex_color)

        # Draw color box
        cv2.rectangle(canvas, (legend_x, y_offset), (legend_x + 30, y_offset + 30),
                     color_bgr, -1)
        cv2.rectangle(canvas, (legend_x, y_offset), (legend_x + 30, y_offset + 30),
                     (0, 0, 0), 1)

        # Draw text
        cv2.putText(canvas, f"Region {region_num}", (legend_x + 40, y_offset + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

        y_offset += 40

    # Add model name at bottom
    cv2.putText(canvas, f"Model: {model_name}", (legend_x, h - 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)

    return canvas

def process_with_easyocr(image_path, reader):
    """Process image with EasyOCR"""
    img = cv2.imread(image_path)
    results = reader.readtext(image_path)

    detections = []
    for bbox, text, conf in results:
        text = text.strip()
        if text.isdigit():
            num = int(text)
            if 0 <= num <= 10:
                detections.append((num, bbox))

    output_img = draw_bounding_boxes(img, detections, "EasyOCR")
    return output_img

def process_with_tesseract(image_path):
    """Process image with Tesseract"""
    img = cv2.imread(image_path)

    # Use Tesseract to get bounding boxes
    data = pytesseract.image_to_data(img, config='--psm 6 digits', output_type=pytesseract.Output.DICT)

    detections = []
    for i, text in enumerate(data['text']):
        if text.strip().isdigit():
            num = int(text.strip())
            if 0 <= num <= 10:
                x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                detections.append((num, (x, y, w, h)))

    output_img = draw_bounding_boxes(img, detections, "Tesseract")
    return output_img

def process_with_paddleocr(image_path, ocr):
    """Process image with PaddleOCR"""
    img = cv2.imread(image_path)
    results = ocr.ocr(image_path, cls=False)

    detections = []
    if results and results[0]:
        for line in results[0]:
            bbox, (text, conf) = line
            text = text.strip()
            if text.isdigit():
                num = int(text)
                if 0 <= num <= 10:
                    detections.append((num, bbox))

    output_img = draw_bounding_boxes(img, detections, "PaddleOCR")
    return output_img

def main():
    """Main processing loop"""
    print("=" * 60)
    print("OCR Comparison Tool - Numbers 0-10")
    print("=" * 60)

    # Check for images
    images = get_images_from_folder('input')

    if images is None or len(images) == 0:
        print("\n❌ ERROR: No images found in 'input' folder!")
        print("   Please add images to the 'input' folder and try again.")
        return

    print(f"\n✓ Found {len(images)} images to process\n")

    # Initialize models
    print("Initializing OCR models...")
    easy_reader = easyocr.Reader(['en'], gpu=False)
    paddle_ocr = PaddleOCR(use_angle_cls=False, lang='en', show_log=False)
    print("✓ Models loaded\n")

    # Process each image
    for idx, image_path in enumerate(images, 1):
        filename = os.path.basename(image_path)
        print(f"[{idx}/{len(images)}] Processing: {filename}")

        # Process with each model
        try:
            # EasyOCR
            result = process_with_easyocr(image_path, easy_reader)
            cv2.imwrite(f'output/easyocr/{filename}', result)
            print(f"  ✓ EasyOCR -> output/easyocr/{filename}")

            # Tesseract
            result = process_with_tesseract(image_path)
            cv2.imwrite(f'output/tesseract/{filename}', result)
            print(f"  ✓ Tesseract -> output/tesseract/{filename}")

            # PaddleOCR
            result = process_with_paddleocr(image_path, paddle_ocr)
            cv2.imwrite(f'output/paddleocr/{filename}', result)
            print(f"  ✓ PaddleOCR -> output/paddleocr/{filename}")

        except Exception as e:
            print(f"  ❌ Error processing {filename}: {e}")

    print("\n" + "=" * 60)
    print("Processing complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
