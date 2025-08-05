#!/usr/bin/env python3
"""
Example usage of the clothing selector program.
This script demonstrates how to use the ClothingSelector class.
"""

import cv2
import numpy as np
from clothing_selector import ClothingSelector

def create_sample_image():
    """Create a simple sample image with colored rectangles representing clothing"""
    # Create a blank image
    img = np.ones((400, 600, 3), dtype=np.uint8) * 255  # White background
    
    # Draw some colored rectangles to represent clothing items
    # Dark shirt
    cv2.rectangle(img, (100, 50), (200, 150), (50, 50, 50), -1)
    
    # Blue jeans
    cv2.rectangle(img, (100, 150), (200, 300), (139, 69, 19), -1)
    
    # Red jacket
    cv2.rectangle(img, (250, 50), (350, 200), (0, 0, 200), -1)
    
    # White shoes
    cv2.rectangle(img, (100, 300), (160, 350), (240, 240, 240), -1)
    cv2.rectangle(img, (180, 300), (240, 350), (240, 240, 240), -1)
    
    # Add some text
    cv2.putText(img, "Sample Clothing Image", (150, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    return img

def main():
    print("Creating a sample image...")
    
    # Create and save a sample image
    sample_img = create_sample_image()
    sample_path = "sample_clothing.jpg"
    cv2.imwrite(sample_path, sample_img)
    print(f"Sample image saved as: {sample_path}")
    
    # Initialize the clothing selector
    print("\nInitializing clothing selector...")
    selector = ClothingSelector()
    
    # Process the sample image
    print(f"\nProcessing sample image...")
    detections = selector.select_clothing(sample_path, "output_sample.jpg")
    
    if detections:
        print(f"\nDetection results:")
        for i, detection in enumerate(detections, 1):
            print(f"{i}. {detection['label']} - Confidence: {detection['confidence']:.2f}")
    else:
        print("No clothing items detected.")
    
    print("\nExample completed!")
    print("You can now use the clothing selector with your own images:")
    print("python3 clothing_selector.py your_image.jpg")
    print("python3 clothing_selector.py your_image.jpg --output result.jpg")

if __name__ == "__main__":
    main()