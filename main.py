"""
Simple OCR comparison tool for numbers 0-10
Compares EasyOCR, Tesseract, and PaddleOCR
"""

import cv2
import numpy as np
import os

import easyocr

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

    # Scale legend based on image size
    legend_width = max(250, int(w * 0.15))
    canvas = np.ones((h, w + legend_width, 3), dtype=np.uint8) * 255
    canvas[:h, :w] = img

    # Draw bounding boxes
    for number, bbox in detections:
        # Get color based on number (cycle through 4 colors)
        color_key = ((number - 1) % 4) + 1
        color_hex = COLORS[color_key]
        color_bgr = hex_to_bgr(color_hex)

        # Draw bounding box - handle different formats
        try:
            if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                # Check if it's polygon format (list of points)
                if isinstance(bbox[0], (list, tuple)) and len(bbox[0]) == 2:
                    # Polygon format: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                    pts = np.array(bbox, dtype=np.int32)
                    cv2.polylines(canvas, [pts], True, color_bgr, 3)
                    label_x, label_y = int(bbox[0][0]), int(bbox[0][1])
                else:
                    # Rectangle format: (x, y, w, h)
                    x, y, bw, bh = bbox
                    cv2.rectangle(canvas, (x, y), (x + bw, y + bh), color_bgr, 3)
                    label_x, label_y = x, y

                # Draw number label
                cv2.putText(canvas, str(number), (label_x, label_y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, color_bgr, 3)
        except Exception as e:
            print(f"    Warning: Failed to draw bbox for number {number}: {e}")

    # Draw legend with scaled fonts
    legend_x = w + 20
    font_scale_title = max(0.8, h / 800)
    font_scale_text = max(0.6, h / 1000)

    cv2.putText(canvas, "Legend", (legend_x, 50),
               cv2.FONT_HERSHEY_SIMPLEX, font_scale_title, (0, 0, 0), 2)

    y_offset = 100
    box_size = max(40, int(h * 0.04))
    spacing = max(60, int(h * 0.06))

    for region_num, hex_color in COLORS.items():
        color_bgr = hex_to_bgr(hex_color)

        # Draw color box
        cv2.rectangle(canvas, (legend_x, y_offset),
                     (legend_x + box_size, y_offset + box_size),
                     color_bgr, -1)
        cv2.rectangle(canvas, (legend_x, y_offset),
                     (legend_x + box_size, y_offset + box_size),
                     (0, 0, 0), 2)

        # Draw text
        cv2.putText(canvas, f"Region {region_num}",
                   (legend_x + box_size + 15, y_offset + box_size // 2 + 5),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale_text, (0, 0, 0), 2)

        y_offset += spacing

    # Add model name at bottom
    cv2.putText(canvas, f"Model: {model_name}", (legend_x, h - 30),
               cv2.FONT_HERSHEY_SIMPLEX, font_scale_text, (100, 100, 100), 2)

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

    print(f"    EasyOCR found {len(detections)} numbers")
    output_img = draw_bounding_boxes(img, detections, "EasyOCR")
    return output_img


def process_with_paddleocr(image_path, ocr):
    """Process image with PaddleOCR"""
    img = cv2.imread(image_path)
    results = ocr.ocr(image_path)

    detections = []
    if results and results[0]:
        for line in results[0]:
            # PaddleOCR returns: [bbox, (text, confidence)]
            if len(line) == 2:
                bbox = line[0]
                text_info = line[1]
                if isinstance(text_info, (list, tuple)) and len(text_info) == 2:
                    text, conf = text_info
                    text = text.strip()
                    if text.isdigit():
                        num = int(text)
                        if 0 <= num <= 10:
                            detections.append((num, bbox))

    print(f"    PaddleOCR found {len(detections)} numbers")
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
    paddle_ocr = PaddleOCR(use_textline_orientation=False, lang='en')
    print("✓ Models loaded\n")

    # Process each image
    for idx, image_path in enumerate(images, 1):
        filename = os.path.basename(image_path)
        print(f"\n[{idx}/{len(images)}] Processing: {filename}")

        # Process with EasyOCR
        try:
            print("  Processing with EasyOCR...")
            result = process_with_easyocr(image_path, easy_reader)
            cv2.imwrite(f'output/easyocr/{filename}', result)
            print(f"  ✓ EasyOCR -> output/easyocr/{filename}")
        except Exception as e:
            print(f"  ❌ EasyOCR error: {e}")


        # Process with PaddleOCR
        try:
            print("  Processing with PaddleOCR...")
            result = process_with_paddleocr(image_path, paddle_ocr)
            cv2.imwrite(f'output/paddleocr/{filename}', result)
            print(f"  ✓ PaddleOCR -> output/paddleocr/{filename}")
        except Exception as e:
            print(f"  ❌ PaddleOCR error: {e}")

    print("\n" + "=" * 60)
    print("Processing complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
