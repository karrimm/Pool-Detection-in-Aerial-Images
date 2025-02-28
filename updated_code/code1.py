import cv2
import numpy as np
import os
import sys

# Configuration dictionary for tunable parameters
config = {
    'hue_lower': 80,      # Lower bound hue in HSV space (blue pools)
    'hue_upper': 140,     # Upper bound hue in HSV space
    'sat_lower': 80,      # Lower bound saturation in HSV space
    'sat_upper': 255,     # Upper bound saturation in HSV space
    'value_lower': 80,    # Lower bound value in HSV space (adjusted for light blues)
    'value_upper': 255,   # Upper bound value in HSV space
    'min_pool_area': 500, # Minimum area threshold to filter small noise (adjustable)
    'remove_duplicates': True,  # Flag to remove nested or overlapping detections
}

def calc_box_limit(points):
    x, y, w, h = points
    centroid_width = x + w // 2
    centroid_height = y + h // 2
    width_offset = w // 2
    height_offset = h // 2
    
    box_limits = (centroid_width - width_offset,
                  centroid_width + width_offset,
                  centroid_height - height_offset,
                  centroid_height + height_offset)
    
    return box_limits

def return_duplicate_indices(box_limits):
    discard = []
    for index, boundary in enumerate(box_limits):
        for outer_index, outer_boundary in enumerate(box_limits):
            if outer_index != index:
                if (boundary[0] >= outer_boundary[0] and
                    boundary[1] <= outer_boundary[1] and
                    boundary[2] >= outer_boundary[2] and
                    boundary[3] <= outer_boundary[3]):
                    print(f'Found duplicate! {boundary} inside of {outer_boundary}')
                    discard.append(index)
    return list(set(discard))  

def detect_swimming_pool(image_path, output_image_path="output_image.jpg", coords_output_path="coordinates.txt"):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Could not load the image.")

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([config['hue_lower'], config['sat_lower'], config['value_lower']])
    upper_blue = np.array([config['hue_upper'], config['sat_upper'], config['value_upper']])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    kernel = np.ones((7, 7), np.uint8)  
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  

    blurred = cv2.GaussianBlur(mask, (5, 5), 0)  

    contours, _ = cv2.findContours(blurred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    pool_contours = []
    original_detections = []  
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > config['min_pool_area']:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h if h != 0 else 0 
            solidity = area / cv2.contourArea(cv2.convexHull(contour)) if area > 0 else 0

            if 0.2 <= aspect_ratio <= 5.0 and solidity > 0.4:  
                pool_contours.append(contour)
                original_detections.append((x, y, w, h))

    if not pool_contours:
        print("No swimming pools detected.")
        return

    if config['remove_duplicates'] and original_detections:
        box_limits = list(map(calc_box_limit, original_detections))
        discard_indices = return_duplicate_indices(box_limits)
        final_contours = [contour for i, contour in enumerate(pool_contours) if i not in discard_indices]
    else:
        final_contours = pool_contours

    if not final_contours:
        print("No swimming pools detected after duplicate removal.")
        return

    output_image = image.copy()  
    all_coords = []  
    for i, contour in enumerate(final_contours):
        cv2.drawContours(output_image, [contour], -1, (0, 0, 255), 2)  

        coords = contour.reshape(-1, 2)  
        all_coords.append(coords)
        coords_file = f"coordinates_pool_{i}.txt"
        with open(coords_file, 'w') as f:
            f.write("x,y\n")
            for x, y in coords:
                f.write(f"{x},{y}\n")
        print(f"Coordinates for pool {i} saved to {coords_file}")

    with open(coords_output_path, 'w') as f:
        f.write("Pool_ID,x,y\n")
        for i, coords in enumerate(all_coords):
            for x, y in coords:
                f.write(f"{i},{x},{y}\n")
    print(f"All coordinates saved to {coords_output_path}")

    cv2.imwrite(output_image_path, output_image)
    print(f"Output image saved as {output_image_path}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python script_name.py <image_path>")
        sys.exit(1)

    input_image = sys.argv[1]

    if not os.path.exists(input_image):
        print(f"Error: Image file '{input_image}' does not exist.")
        sys.exit(1)

    try:
        detect_swimming_pool(input_image)
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()

