# OCR Comparison Tool

Compare 3 OCR models (EasyOCR, Tesseract, PaddleOCR) for detecting numbers 0-10.

## Project Structure

```
ocr/
├── input/              # Put your images here
├── output/
│   ├── easyocr/       # EasyOCR results
│   ├── tesseract/     # Tesseract results
│   └── paddleocr/     # PaddleOCR results
├── main.py
└── requirements.txt
```

## Color Coding

Numbers are color-coded with these regions:
- Region 1: #D94E4E (warm muted red)
- Region 2: #3A77C2 (deep denim blue)
- Region 3: #E7C85A (soft golden yellow)
- Region 4: #4FA36E (medium jade green)

## Installation

```bash
pip install -r requirements.txt
```

**Note**: Tesseract requires a system binary installation:
- **Windows**: Download from https://github.com/UB-Mannheim/tesseract/wiki
- **Linux**: `sudo apt-get install tesseract-ocr`
- **Mac**: `brew install tesseract`

## Configuration

Edit `main.py` to set the target number color:

```python
TARGET_NUMBER_COLOR = '#3A77C2'  # Hex color of numbers (e.g., blue)
COLOR_TOLERANCE = 30              # HSV tolerance (higher = more colors)
```

Set `TARGET_NUMBER_COLOR = None` to disable color extraction.

## Usage

1. Put your images in the `input/` folder
2. Configure the hex color of your numbers in `main.py`
3. Run: `python main.py`
4. Check results in:
   - `output/easyocr/` - EasyOCR results
   - `output/paddleocr/` - PaddleOCR results
   - `output/preprocessed/` - Extracted color masks (for debugging)

Each output image shows:
- Original image with bounding boxes
- Color-coded numbers
- Legend on the right side
