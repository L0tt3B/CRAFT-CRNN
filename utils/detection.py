import cv2
import numpy as np
import os
import shutil

def get_word_boxes_fixed_padding(
    region_map, affinity_map,
    region_thr=0.4, affinity_thr=0.2,
    pad_x=3,  # horizontal padding in pixels (left & right)
    pad_y=4   # vertical padding in pixels (top & bottom)
):
    """
    1. Threshold region & affinity maps
    2. Merge => connected components for entire words
    3. For each connected component, get boundingRect
    4. Expand boundingRect by a fixed pixel margin horizontally & vertically
       (left/right => pad_x, top/bottom => pad_y).
    """
    # 1) Threshold region & affinity
    region_bin = (region_map > region_thr).astype(np.uint8)
    affinity_bin = (affinity_map > affinity_thr).astype(np.uint8)
    combined = np.clip(region_bin + affinity_bin, 0, 1).astype(np.uint8)

    # 2) Find contours in the merged map
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    h, w = region_map.shape[:2]
    boxes = []
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)

        # Expand bounding box by fixed pixels
        x_padded = x - pad_x
        y_padded = y - pad_y
        bw_padded = bw + 2 * pad_x
        bh_padded = bh + 2 * pad_y

        # Clamp so we don't go outside the model's output map
        if x_padded < 0:
            bw_padded += x_padded
            x_padded = 0
        if y_padded < 0:
            bh_padded += y_padded
            y_padded = 0
        if x_padded + bw_padded > w:
            bw_padded = w - x_padded
        if y_padded + bh_padded > h:
            bh_padded = h - y_padded

        boxes.append((x_padded, y_padded, bw_padded, bh_padded))

    return boxes

def group_boxes_by_lines(boxes, vertical_threshold=20):
    # Calculate the vertical center
    box_centers = [(i, box[1] + box[3] / 2.0) for i, box in enumerate(boxes)]
    # Sort boxes by vertical center.
    box_centers.sort(key=lambda x: x[1])
    
    lines = []
    current_line = [box_centers[0][0]]
    current_center = box_centers[0][1]
    
    for i, center in box_centers[1:]:
        # group by y
        if abs(center - current_center) <= vertical_threshold:
            current_line.append(i)
            current_center = np.mean([boxes[idx][1] + boxes[idx][3]/2.0 for idx in current_line])
        else:
            lines.append(current_line)
            current_line = [i]
            current_center = center
    if current_line:
        lines.append(current_line)
    
    # sort by x
    sorted_boxes = []
    for line in lines:
        line_boxes = [boxes[idx] for idx in line]
        line_boxes.sort(key=lambda b: b[0])
        sorted_boxes.extend(line_boxes)
        
    return sorted_boxes

def clear_and_create_demo_folder(demo_folder='../demo_image'):
    os.makedirs(demo_folder, exist_ok=True)

    # Delete all existing files
    for filename in os.listdir(demo_folder):
        file_path = os.path.join(demo_folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")

if __name__ == "__main__":
    print("Detection utilities loaded successfully")
