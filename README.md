# Clothing Selector

A Python program that can detect and select clothing items in images using computer vision techniques.

## Features

- **YOLO Object Detection**: Uses YOLOv3 for accurate object detection (when model files are available)
- **Contour Detection**: Falls back to color-based segmentation and contour detection
- **Multiple Clothing Types**: Detects various clothing items including shirts, pants, accessories
- **Visual Output**: Draws bounding boxes around detected clothing items
- **Batch Processing**: Can process multiple images

## Installation

The required dependencies are already installed:
- OpenCV (cv2)
- NumPy

## Usage

### Basic Usage

Process a single image:
```bash
python3 clothing_selector.py your_image.jpg
```

Save the result to a file:
```bash
python3 clothing_selector.py your_image.jpg --output result.jpg
```

### Example Usage

Run the example script to see how it works:
```bash
python3 example_usage.py
```

This will create a sample image and demonstrate the clothing detection.

### Programmatic Usage

```python
from clothing_selector import ClothingSelector

# Initialize the selector
selector = ClothingSelector()

# Process an image
detections = selector.select_clothing("your_image.jpg", "output.jpg")

# Print results
for detection in detections:
    print(f"Found {detection['label']} with confidence {detection['confidence']:.2f}")
```

## How It Works

1. **YOLO Detection**: If YOLO model files are available, the program uses YOLOv3 to detect objects in the image, filtering for clothing-related items.

2. **Contour Detection**: If YOLO is not available, the program falls back to:
   - Converting the image to HSV color space
   - Creating masks for common clothing colors
   - Finding contours in the masked regions
   - Filtering by size and aspect ratio

3. **Visualization**: The program draws bounding boxes around detected clothing items and labels them with confidence scores.

## Supported Clothing Items

When using YOLO detection:
- Person (for full body detection)
- Handbag
- Tie
- Suitcase
- Backpack
- Umbrella

When using contour detection:
- Any clothing region based on color segmentation

## Optional: Enhanced YOLO Detection

For better accuracy, you can download YOLO model files:

```bash
wget https://pjreddie.com/media/files/yolov3.weights
wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg
```

## Files Created

- `clothing_selector.py` - Main program
- `example_usage.py` - Example demonstration
- `sample_clothing.jpg` - Sample image created by example
- `output_sample.jpg` - Result with detected clothing regions
- `requirements.txt` - Dependencies list
- `README.md` - This documentation

## Limitations

- The basic contour detection method works best with solid-colored clothing
- YOLO detection requires downloading additional model files (~250MB)
- Detection accuracy depends on image quality and lighting conditions
- Currently optimized for common clothing colors
