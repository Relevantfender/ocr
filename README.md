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

## Usage

1. Put your images in the `input/` folder
2. Run: `python main.py`
3. Check results in `output/easyocr/`, `output/tesseract/`, and `output/paddleocr/`

Each output image shows:
- Original image with bounding boxes
- Color-coded numbers
- Legend on the right side
