"""
Simple OCR tool for numbers 0-10 using EasyOCR
Extracts specific color, inverts to white-on-black for best results
"""

import cv2
import numpy as np
import os
import easyocr

# ===== CONFIGURATION =====
# Set the hex color of the numbers you want to extract
TARGET_NUMBER_COLOR = '#1a6db3'  # Blue - change this to match your numbers
COLOR_TOLERANCE = 30  # HSV tolerance (higher = more colors matched)

# GPU Settings (requires CUDA and GPU version of pytorch)
USE_GPU = False  # Set to True to use GPU (much faster)
# =========================

# Color mapping for numbers (will be used later for floodFill)
COLORS = {
    1: '#D94E4E',  # warm muted red
    2: '#3A77C2',  # deep denim blue
    3: '#E7C85A',  # soft golden yellow
    4: '#4FA36E',  # medium jade green
    5: '#8B5CF6',  # purple
    6: '#EC4899',  # pink
}

def hex_to_bgr(hex_color):
    """Convert hex color to BGR for OpenCV"""
    hex_color = hex_color.lstrip('#')
    rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return (rgb[2], rgb[1], rgb[0])  # Convert RGB to BGR

def flood_fill_numbers(image, detections, target_color_hex):
    """
    Flood fill detected number regions with their corresponding colors

    Args:
        image: Original image (BGR)
        detections: List of (number, bbox) tuples from OCR
        target_color_hex: Hex color of the numbers to find (e.g., '#1a6db3')

    Returns:
        Image with flood-filled colored regions
    """
    h, w = image.shape[:2]

    # Convert target color to BGR and HSV
    target_bgr = hex_to_bgr(target_color_hex)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    target_hsv = cv2.cvtColor(np.uint8([[target_bgr]]), cv2.COLOR_BGR2HSV)[0][0]

    print(f"    Creating flood fill masks for {len(detections)} numbers...")

    # Step 1: Create work image - convert ALL blue pixels to white (do this once)
    print("    Converting all blue pixels to white...")
    work_img = image.copy()
    for y in range(h):
        for x in range(w):
            pixel_hsv = hsv[y, x]
            h_diff = abs(int(pixel_hsv[0]) - int(target_hsv[0]))
            if h_diff < 15 and pixel_hsv[1] > 50 and pixel_hsv[2] > 50:
                work_img[y, x] = [255, 255, 255]

    # List to store (mask, color) tuples
    masks_to_apply = []

    # Step 2: For each number, flood fill its connected white component
    for num, bbox in detections:
        # Get fill color for this number
        color_key = ((num - 1) % 6) + 1
        fill_color = hex_to_bgr(COLORS[color_key])

        # Get bounding box coordinates and center
        pts = np.array(bbox, dtype=np.int32)
        x_coords = [pt[0] for pt in pts]
        y_coords = [pt[1] for pt in pts]
        min_x, max_x = int(min(x_coords)), int(max(x_coords))
        min_y, max_y = int(min(y_coords)), int(max(y_coords))

        # Use center of bounding box as seed point
        center_x = (min_x + max_x) // 2
        center_y = (min_y + max_y) // 2
        seed_point = (center_x, center_y)
        print(f"      Number '{num}': Using bbox center as seed at {seed_point}")

        # Check if seed point is on white pixel (it should be)
        if work_img[center_y, center_x][0] < 200:
            print(f"      WARNING: Seed point for '{num}' not on white region, skipping")
            continue

        # Flood fill this connected white component
        mask = np.zeros((h + 2, w + 2), np.uint8)
        work_copy = work_img.copy()
        lo_diff = (10, 10, 10)  # Small tolerance - fill connected white pixels
        up_diff = (10, 10, 10)
        flags = 4 | cv2.FLOODFILL_MASK_ONLY | (255 << 8)

        cv2.floodFill(work_copy, mask, seed_point, (0, 0, 0), lo_diff, up_diff, flags)
        mask_region = mask[1:-1, 1:-1]

        filled_pixels = np.sum(mask_region == 255)
        print(f"      Mask for '{num}': {filled_pixels} pixels filled")

        # Store mask and color
        masks_to_apply.append((mask_region, fill_color, num))

    # NOW apply all masks to create final image
    print(f"    Applying {len(masks_to_apply)} masks to fresh white canvas...")

    # Step 3: Start with a pure white canvas
    filled_img = np.ones((h, w, 3), dtype=np.uint8) * 255

    # Step 4: Apply all colored masks
    pixels_changed = 0
    for mask_region, fill_color, num in masks_to_apply:
        mask_pixels = np.sum(mask_region == 255)
        filled_img[mask_region == 255] = fill_color
        pixels_changed += mask_pixels
        print(f"      Applied color {fill_color} for number '{num}' ({mask_pixels} pixels)")

    # Step 5: Copy boundary pixels - exclude white and blue, keep only dark boundaries
    print("    Copying boundary pixels from original...")

    # Create mask excluding white pixels (background)
    not_white = np.any(image < 200, axis=2)

    # Create mask excluding blue pixels (the numbers we converted)
    not_blue = np.zeros((h, w), dtype=bool)
    for y in range(h):
        for x in range(w):
            pixel_hsv = hsv[y, x]
            h_diff = abs(int(pixel_hsv[0]) - int(target_hsv[0]))
            is_blue = (h_diff < 15 and pixel_hsv[1] > 50 and pixel_hsv[2] > 50)
            not_blue[y, x] = not is_blue

    # Combine: pixels that are neither white nor blue = boundaries
    boundary_mask = not_white & not_blue
    filled_img[boundary_mask] = image[boundary_mask]
    boundary_pixels = np.sum(boundary_mask)
    print(f"    Copied {boundary_pixels} boundary pixels from original")

    print(f"    ✓ Flood fill complete! Total colored pixels: {pixels_changed}")
    return filled_img

def create_side_by_side_output(original, filled, title="OCR Result"):
    """
    Create side-by-side comparison of original and flood-filled images with legend

    Args:
        original: Original image
        filled: Flood-filled image
        title: Title for the output

    Returns:
        Side-by-side image with legend
    """
    h, w = original.shape[:2]

    # Calculate legend width based on image size
    legend_width = max(250, int(w * 0.15))

    # Create canvas: original + filled + legend
    canvas = np.ones((h, w * 2 + legend_width, 3), dtype=np.uint8) * 255

    # Place images
    canvas[:h, :w] = original
    canvas[:h, w:w*2] = filled

    # Add labels
    font_scale = max(0.7, h / 1000)
    cv2.putText(canvas, "Original", (10, 40),
               cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 2)
    cv2.putText(canvas, "Flood Filled", (w + 10, 40),
               cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 2)

    # Draw legend on the right
    legend_x = w * 2 + 20
    legend_y = 80

    # Legend title
    cv2.putText(canvas, "Legend", (legend_x, 50),
               cv2.FONT_HERSHEY_SIMPLEX, font_scale * 1.2, (0, 0, 0), 2)

    # Draw color boxes for each region
    box_size = max(40, int(h * 0.04))
    spacing = max(70, int(h * 0.06))

    for region_num, hex_color in COLORS.items():
        color_bgr = hex_to_bgr(hex_color)

        # Draw filled color box
        cv2.rectangle(canvas,
                     (legend_x, legend_y),
                     (legend_x + box_size, legend_y + box_size),
                     color_bgr, -1)

        # Draw border
        cv2.rectangle(canvas,
                     (legend_x, legend_y),
                     (legend_x + box_size, legend_y + box_size),
                     (0, 0, 0), 2)

        # Add region text
        text = f"Region {region_num}"
        cv2.putText(canvas, text,
                   (legend_x + box_size + 10, legend_y + box_size // 2 + 5),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.7, (0, 0, 0), 2)

        # Add hex color code below
        cv2.putText(canvas, hex_color,
                   (legend_x + box_size + 10, legend_y + box_size // 2 + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.5, (100, 100, 100), 1)

        legend_y += spacing

    # Add note about number cycling
    note_y = legend_y + 20
    cv2.putText(canvas, "Numbers cycle", (legend_x, note_y),
               cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.6, (100, 100, 100), 1)
    cv2.putText(canvas, "through regions", (legend_x, note_y + 20),
               cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.6, (100, 100, 100), 1)
    cv2.putText(canvas, "1-6 repeatedly", (legend_x, note_y + 40),
               cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.6, (100, 100, 100), 1)

    return canvas

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

def extract_color(image, hex_color, tolerance=30):
    """
    Extract specific hex color from image and create clean binary mask
    GUARANTEES output has EXACT same dimensions as input - no resizing!

    Args:
        image: Input image (BGR)
        hex_color: Hex color string (e.g., '#1a6db3')
        tolerance: HSV tolerance for color matching (default: 30)

    Returns:
        Preprocessed image with BLACK numbers on WHITE background, same size as input
    """
    # Get original dimensions - we will NOT change these
    h, w = image.shape[:2]

    # Convert hex to BGR
    target_bgr = hex_to_bgr(hex_color)

    # Convert to HSV for better color matching
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    target_hsv = cv2.cvtColor(np.uint8([[target_bgr]]), cv2.COLOR_BGR2HSV)[0][0]

    # Create HSV range with tolerance
    lower = np.array([max(0, target_hsv[0] - tolerance), 50, 50])
    upper = np.array([min(179, target_hsv[0] + tolerance), 255, 255])

    # Create mask for the specific color
    mask = cv2.inRange(hsv, lower, upper)

    # Clean up noise with minimal kernel to keep numbers sharp
    # Using (2,2) kernel to avoid any edge effects that could shift pixels
    kernel = np.ones((2, 2), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Invert so we get BLACK numbers on WHITE background (standard for OCR)
    mask = cv2.bitwise_not(mask)

    # Verify dimensions haven't changed (they shouldn't, but safety check)
    assert mask.shape[:2] == (h, w), f"Dimension mismatch! Input: {(h,w)}, Output: {mask.shape[:2]}"

    return mask

def draw_bounding_boxes(image, detections, model_name, preprocessed=None):
    """
    Draw bounding boxes on image with color-coded numbers and legend

    Args:
        image: Original image
        detections: List of (number, bbox) tuples
        model_name: Name of the OCR model
        preprocessed: Optional preprocessed/filtered image to show

    Returns:
        Image with bounding boxes and legend
    """
    img = image.copy()
    h, w = img.shape[:2]

    # If preprocessed image provided, show it alongside
    if preprocessed is not None:
        # Convert grayscale to BGR if needed
        if len(preprocessed.shape) == 2:
            preprocessed_bgr = cv2.cvtColor(preprocessed, cv2.COLOR_GRAY2BGR)
        else:
            preprocessed_bgr = preprocessed

        # CRITICAL: Preprocessed MUST be same dimensions as original
        # If not, something went wrong in extract_color()
        prep_h, prep_w = preprocessed_bgr.shape[:2]
        if prep_h != h or prep_w != w:
            print(f"WARNING: Dimension mismatch! Original: {(h,w)}, Preprocessed: {(prep_h,prep_w)}")
            # Force resize to match (shouldn't be needed if extract_color works correctly)
            preprocessed_bgr = cv2.resize(preprocessed_bgr, (w, h))

        prep_width = w  # Same width as original

        # Add label to preprocessed
        label_img = preprocessed_bgr.copy()
        cv2.putText(label_img, "Filtered", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Create canvas with space for: preprocessed + original + legend
        legend_width = max(250, int(w * 0.15))
        total_width = prep_width + w + legend_width
        canvas = np.ones((h, total_width, 3), dtype=np.uint8) * 255
        canvas[:h, :prep_width] = label_img
        canvas[:h, prep_width:prep_width + w] = img

        # Bounding boxes from preprocessed are at coords (0 to w, 0 to h)
        # On filtered side: draw at original coords
        # On original side: draw at coords + prep_width offset
        bbox_offset = prep_width
        legend_x = prep_width + w + 20
    else:
        # Scale legend based on image size
        legend_width = max(250, int(w * 0.15))
        canvas = np.ones((h, w + legend_width, 3), dtype=np.uint8) * 255
        canvas[:h, :w] = img
        bbox_offset = 0
        legend_x = w + 20

    # Draw bounding boxes
    for number, bbox in detections:
        # Get color based on number (cycle through 6 colors)
        color_key = ((number - 1) % 6) + 1
        color_hex = COLORS[color_key]
        color_bgr = hex_to_bgr(color_hex)

        # Draw bounding box - handle different formats
        try:
            if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                # Check if it's polygon format (list of points)
                if isinstance(bbox[0], (list, tuple)) and len(bbox[0]) == 2:
                    # Polygon format: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                    pts = np.array(bbox, dtype=np.int32)

                    # Draw on filtered image (if shown) at original coordinates
                    if preprocessed is not None:
                        cv2.polylines(canvas, [pts], True, color_bgr, 2)

                    # Draw on original image with offset
                    pts_offset = pts.copy()
                    pts_offset[:, 0] += bbox_offset
                    cv2.polylines(canvas, [pts_offset], True, color_bgr, 3)
                    label_x, label_y = int(bbox[0][0]) + bbox_offset, int(bbox[0][1])
                else:
                    # Rectangle format: (x, y, w, h)
                    x, y, bw, bh = bbox

                    # Draw on filtered image (if shown)
                    if preprocessed is not None:
                        cv2.rectangle(canvas, (x, y), (x + bw, y + bh), color_bgr, 2)

                    # Draw on original image with offset
                    cv2.rectangle(canvas, (x + bbox_offset, y), (x + bbox_offset + bw, y + bh), color_bgr, 3)
                    label_x, label_y = x + bbox_offset, y

                # Draw number label
                cv2.putText(canvas, str(number), (label_x, label_y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, color_bgr, 3)
        except Exception as e:
            print(f"    Warning: Failed to draw bbox for number {number}: {e}")

    # Draw legend with scaled fonts
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
    """Process image with EasyOCR using inverted polarity (white-on-black)"""
    img = cv2.imread(image_path)
    preprocessed = None
    detections = []

    # Preprocess if color extraction is enabled
    if TARGET_NUMBER_COLOR:
        # Extract color
        preprocessed = extract_color(img, TARGET_NUMBER_COLOR, COLOR_TOLERANCE)

        # Invert to white-on-black (works best for EasyOCR)
        preprocessed = cv2.bitwise_not(preprocessed)

        # Process with EasyOCR
        results = reader.readtext(preprocessed)
        for bbox, text, conf in results:
            text = text.strip()
            if text.isdigit():
                num = int(text)
                if 0 <= num <= 10:
                    detections.append((num, bbox))
    else:
        # No color extraction, process original
        results = reader.readtext(image_path)
        for bbox, text, conf in results:
            text = text.strip()
            if text.isdigit():
                num = int(text)
                if 0 <= num <= 10:
                    detections.append((num, bbox))

    print(f"    EasyOCR found {len(detections)} numbers")

    # Flood fill the detected numbers with their colors
    if TARGET_NUMBER_COLOR and len(detections) > 0:
        filled_img = flood_fill_numbers(img, detections, TARGET_NUMBER_COLOR)
        output_img = create_side_by_side_output(img, filled_img)
    else:
        # If no color extraction or no detections, just show original with bounding boxes
        output_img = draw_bounding_boxes(img, detections, "EasyOCR", preprocessed)

    return output_img


def main():
    """Main processing loop"""
    print("=" * 60)
    print("OCR Tool - Numbers 0-10 (EasyOCR)")
    print("=" * 60)

    # Check for images
    images = get_images_from_folder('input')

    if images is None or len(images) == 0:
        print("\n❌ ERROR: No images found in 'input' folder!")
        print("   Please add images to the 'input' folder and try again.")
        return

    print(f"\n✓ Found {len(images)} images to process\n")

    # Initialize EasyOCR
    print("Initializing EasyOCR...")
    print(f"GPU mode: {'ENABLED' if USE_GPU else 'DISABLED'}")

    if USE_GPU:
        print("Note: GPU requires proper CUDA setup (PyTorch with CUDA).")

    try:
        easy_reader = easyocr.Reader(['en'], gpu=USE_GPU)
        print("✓ EasyOCR loaded\n")
    except Exception as e:
        print(f"\n❌ Error loading EasyOCR: {e}")
        print("\nIf GPU is enabled and failing:")
        print("  Install PyTorch with CUDA support")
        print("  Or set USE_GPU = False in main.py to use CPU\n")
        return

    # Process each image
    for idx, image_path in enumerate(images, 1):
        filename = os.path.basename(image_path)
        print(f"\n[{idx}/{len(images)}] Processing: {filename}")

        # Process with EasyOCR
        try:
            result = process_with_easyocr(image_path, easy_reader)
            cv2.imwrite(f'output/easyocr/{filename}', result)
            print(f"  ✓ EasyOCR -> output/easyocr/{filename}")
        except Exception as e:
            print(f"  ❌ EasyOCR error: {e}")

    print("\n" + "=" * 60)
    print("Processing complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
