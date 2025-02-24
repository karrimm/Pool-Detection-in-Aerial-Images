import cv2
import numpy as np
import sys
import os

def detect_and_crop_pools(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Could not load the image.")
        return
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    lower_cyan = np.array([75, 45, 100]) 
    upper_cyan = np.array([120, 255, 255]) 
    
    # Create a mask using the defined cyan range
    mask = cv2.inRange(hsv, lower_cyan, upper_cyan)
    
    # Apply morphological operations to reduce noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   
    
    # Find contours in the processed mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours 
    min_area = 500
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    
    output_img = img.copy()
    coordinates = []  # 

    for idx, contour in enumerate(filtered_contours):
        # Approximate the contour to make it smoother
        epsilon = 0.011 * cv2.arcLength(contour, True)  
        approx_contour = cv2.approxPolyDP(contour, epsilon, True)
        
        cv2.drawContours(output_img, [approx_contour], -1, (255, 0, 0), 1)
        
        pool_coords = []
        for point in approx_contour:
            x, y = point[0]
            pool_coords.append((x, y))
        coordinates.append(pool_coords)
    
    with open('coordinates.txt', 'w') as file:
        for idx, pool_coords in enumerate(coordinates):
            file.write(f"Pool {idx + 1} Coordinates:\n")
            for x, y in pool_coords:
                file.write(f"{x},{y}\n")
            file.write("\n") 
    
    output_dir = "cropped_pools"
    os.makedirs(output_dir, exist_ok=True)
    
    for idx, contour in enumerate(filtered_contours):
        x, y, w, h = cv2.boundingRect(contour)
        
        padding = 14 
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(output_img.shape[1] - x, w + 2 * padding)
        h = min(output_img.shape[0] - y, h + 2 * padding)
        
        cropped_pool = output_img[y:y+h, x:x+w]
        
        output_path = os.path.join(output_dir, f"pool_{idx + 1}.jpg")
        cv2.imwrite(output_path, cropped_pool)
    
    cv2.imwrite("output_image_with_smooth_outlines.jpg", output_img)
    print(f"Detected {len(filtered_contours)} pool(s). Cropped images saved in '{output_dir}'.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python pool_detector.py <input_image.jpg>")
        sys.exit(1)
    detect_and_crop_pools(sys.argv[1])
