import cv2
import numpy as np
import argparse
import os
from pathlib import Path

class ClothingSelector:
    def __init__(self):
        # COCO class names - clothing related items
        self.clothing_classes = {
            0: 'person',
            # Common clothing items that might be detected as objects
            26: 'handbag',
            27: 'tie',
            28: 'suitcase',
            31: 'backpack',
            32: 'umbrella',
        }
        
        # Colors for different clothing items (BGR format)
        self.colors = {
            0: (255, 0, 0),    # person - blue
            26: (0, 255, 0),   # handbag - green
            27: (0, 0, 255),   # tie - red
            28: (255, 255, 0), # suitcase - cyan
            31: (255, 0, 255), # backpack - magenta
            32: (0, 255, 255), # umbrella - yellow
        }
        
        # Load YOLO model (you'll need to download these files)
        self.net = None
        self.output_layers = None
        self.load_yolo_model()
    
    def load_yolo_model(self):
        """Load YOLO model for object detection"""
        try:
            # Try to load YOLOv3 model (you'll need to download these)
            weights_path = "yolov3.weights"
            config_path = "yolov3.cfg"
            
            if os.path.exists(weights_path) and os.path.exists(config_path):
                self.net = cv2.dnn.readNet(weights_path, config_path)
                layer_names = self.net.getLayerNames()
                self.output_layers = [layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
                print("YOLO model loaded successfully!")
            else:
                print("YOLO model files not found. Using basic contour detection instead.")
                print("To use YOLO detection, download:")
                print("- yolov3.weights")
                print("- yolov3.cfg")
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            print("Using basic contour detection instead.")
    
    def detect_with_yolo(self, image):
        """Detect objects using YOLO model"""
        if self.net is None:
            return []
        
        height, width, channels = image.shape
        
        # Prepare image for YOLO
        blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layers)
        
        # Process detections
        boxes = []
        confidences = []
        class_ids = []
        
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > 0.5 and class_id in self.clothing_classes:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        # Apply non-maximum suppression
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        
        detections = []
        if len(indexes) > 0:
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                label = self.clothing_classes[class_ids[i]]
                confidence = confidences[i]
                detections.append({
                    'bbox': (x, y, w, h),
                    'label': label,
                    'confidence': confidence,
                    'class_id': class_ids[i]
                })
        
        return detections
    
    def detect_with_contours(self, image):
        """Basic clothing detection using contours and color segmentation"""
        # Convert to different color spaces for better segmentation
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Create masks for common clothing colors
        clothing_regions = []
        
        # Define color ranges for common clothing colors
        color_ranges = [
            # Dark colors (black, navy, dark gray)
            ([0, 0, 0], [180, 255, 50]),
            # Blue colors
            ([100, 50, 50], [130, 255, 255]),
            # Red colors
            ([0, 50, 50], [10, 255, 255]),
            ([170, 50, 50], [180, 255, 255]),
            # White/light colors
            ([0, 0, 200], [180, 30, 255]),
        ]
        
        combined_mask = np.zeros(gray.shape, dtype=np.uint8)
        
        for lower, upper in color_ranges:
            lower = np.array(lower)
            upper = np.array(upper)
            mask = cv2.inRange(hsv, lower, upper)
            combined_mask = cv2.bitwise_or(combined_mask, mask)
        
        # Apply morphological operations to clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area > 1000:  # Filter small areas
                x, y, w, h = cv2.boundingRect(contour)
                # Filter by aspect ratio to focus on clothing-like shapes
                aspect_ratio = w / h
                if 0.3 < aspect_ratio < 3.0:
                    detections.append({
                        'bbox': (x, y, w, h),
                        'label': 'clothing_region',
                        'confidence': 0.7,
                        'class_id': -1
                    })
        
        return detections
    
    def select_clothing(self, image_path, output_path=None):
        """Main function to select clothing in an image"""
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load image from {image_path}")
            return
        
        print(f"Processing image: {image_path}")
        original_image = image.copy()
        
        # Try YOLO detection first, fall back to contour detection
        detections = self.detect_with_yolo(image)
        
        if not detections:
            print("No clothing detected with YOLO, trying contour detection...")
            detections = self.detect_with_contours(image)
        
        if not detections:
            print("No clothing items detected in the image.")
            return
        
        print(f"Found {len(detections)} clothing item(s)")
        
        # Draw bounding boxes and labels
        for detection in detections:
            x, y, w, h = detection['bbox']
            label = detection['label']
            confidence = detection['confidence']
            class_id = detection['class_id']
            
            # Choose color
            color = self.colors.get(class_id, (0, 255, 0))
            
            # Draw bounding box
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            
            # Draw label
            label_text = f"{label}: {confidence:.2f}"
            label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(image, (x, y - label_size[1] - 10), (x + label_size[0], y), color, -1)
            cv2.putText(image, label_text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            print(f"- {label} at ({x}, {y}, {w}, {h}) with confidence {confidence:.2f}")
        
        # Save or display result
        if output_path:
            cv2.imwrite(output_path, image)
            print(f"Result saved to: {output_path}")
        else:
            # Display the image
            cv2.imshow('Clothing Detection', image)
            cv2.imshow('Original', original_image)
            print("Press any key to close the windows...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return detections

def main():
    parser = argparse.ArgumentParser(description='Select clothing items in images')
    parser.add_argument('image_path', help='Path to the input image')
    parser.add_argument('--output', '-o', help='Path to save the output image')
    parser.add_argument('--download-yolo', action='store_true', help='Download YOLO model files')
    
    args = parser.parse_args()
    
    if args.download_yolo:
        print("To download YOLO model files, run:")
        print("wget https://pjreddie.com/media/files/yolov3.weights")
        print("wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg")
        return
    
    if not os.path.exists(args.image_path):
        print(f"Error: Image file '{args.image_path}' not found.")
        return
    
    # Create clothing selector and process image
    selector = ClothingSelector()
    selector.select_clothing(args.image_path, args.output)

if __name__ == "__main__":
    main()