# OCR Number Detection and Segmentation

A computer vision project that detects numbers (0-10), segments them using SAM (Segment Anything Model), and applies color fills to specific regions.

## Project Structure

```
ocr/
├── src/
│   ├── detector.py          # Number detection (0-10 only)
│   ├── segmenter.py         # SAM model integration
│   ├── colorizer.py         # Flood fill with cv2
│   └── pipeline.py          # Main pipeline
├── tests/
│   └── test_images/         # Sample images for testing
├── requirements.txt
└── README.md
```

## Components

### 1. Number Detector
Detects numbers 0-10 in images and returns their bounding box locations.

### 2. SAM Segmenter
Uses Segment Anything Model to segment detected number regions.

### 3. Colorizer
Applies flood fill algorithm to color segmented regions with specific hex colors.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
from src.pipeline import OCRPipeline

pipeline = OCRPipeline()
result = pipeline.process("path/to/image.jpg")
```

## Development Status

- [ ] Number detection module
- [ ] SAM segmentation module
- [ ] Flood fill colorizer
- [ ] Main pipeline integration
