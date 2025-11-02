import matplotlib.pyplot as plt
import cv2
import os
import numpy as np

def visualize_detections(orig_image, boxes, title='Detected Text Regions', save_path=None):
    """
    Visualize detected text regions on the original image
    """
    plt.figure(figsize=(10, 10))
    plt.imshow(orig_image)
    ax = plt.gca()
    
    for (sx, sy, sw, sh) in boxes:
        rect = plt.Rectangle((sx, sy), sw, sh, edgecolor='red', fill=False, linewidth=2)
        ax.add_patch(rect)
    
    plt.title(title)
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    plt.show()

def save_detection_results(orig_image, boxes, output_dir='../result'):
    """
    Save detection results including image with boxes and coordinates
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Scale boxes if needed (assuming boxes are already scaled)
    scaled_boxes = boxes
    
    # Save image with boxes
    result_img = orig_image.copy()
    result_img = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
    for (sx, sy, sw, sh) in scaled_boxes:
        cv2.rectangle(result_img, (sx, sy), (sx+sw, sy+sh), (0, 0, 255), 2)
    
    image_save_path = os.path.join(output_dir, 'detected_image.jpg')
    cv2.imwrite(image_save_path, result_img)
    
    # Save coordinates
    coords_save_path = os.path.join(output_dir, 'coords.txt')
    with open(coords_save_path, 'w') as f:
        for (sx, sy, sw, sh) in scaled_boxes:
            f.write(f"{sx},{sy},{sw},{sh}\n")
    
    # Save binary mask (optional)
    # This would require the region_map which might not be available here
    # You can modify this function to accept region_map if needed
    
    return image_save_path, coords_save_path

if __name__ == "__main__":
    print("Visualization utilities loaded successfully")
