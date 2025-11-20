# OCR Tool for Numbers 0-10

Simple OCR tool using EasyOCR to detect and highlight numbers 0-10 in images.

## Features

- Extracts specific color from images (configurable hex color)
- Inverts to white-on-black for optimal OCR accuracy
- Draws color-coded bounding boxes around detected numbers
- Shows preprocessed image alongside original with legend

## Project Structure

```
ocr/
├── input/              # Put your images here
├── output/
│   └── easyocr/       # EasyOCR results with bounding boxes
├── main.py
└── requirements.txt
```

## Color Coding

Numbers are color-coded by region (cycles through 6 colors):
- Region 1: #D94E4E (warm muted red)
- Region 2: #3A77C2 (deep denim blue)
- Region 3: #E7C85A (soft golden yellow)
- Region 4: #4FA36E (medium jade green)
- Region 5: #8B5CF6 (purple)
- Region 6: #EC4899 (pink)

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

Edit `main.py` to set the target number color:

```python
TARGET_NUMBER_COLOR = '#1a6db3'  # Hex color of numbers to extract
COLOR_TOLERANCE = 30              # HSV tolerance (higher = more colors)
USE_GPU = False                   # Set True for GPU acceleration
```

Set `TARGET_NUMBER_COLOR = None` to disable color extraction and process the original image.

## Usage

1. Put your images in the `input/` folder
2. Configure the hex color of your numbers in `main.py`
3. Run: `python main.py`
4. Check results in `output/easyocr/`

Each output image shows:
- **Left**: Preprocessed image (inverted white-on-black) with bounding boxes
- **Middle**: Original image with bounding boxes
- **Right**: Color legend

## How It Works

1. **Color Extraction**: Extracts only the specified hex color from the image
2. **Inversion**: Converts to white-on-black (optimal for EasyOCR)
3. **OCR Processing**: EasyOCR detects numbers 0-10
4. **Visualization**: Draws bounding boxes on both preprocessed and original images

## GPU Support

For faster processing, install PyTorch with CUDA:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

Then set `USE_GPU = True` in `main.py`.
