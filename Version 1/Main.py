import cv2
import numpy as np
import sys

def detect_pools(image_path):
    # Read the input image
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Could not load the image.")
        return
    
    # Convert image to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    lower_blue = np.array([85, 50, 100])   # Lower bound (H, S, V)
    upper_blue = np.array([120, 255, 255])  # Upper bound (H, S, V)

    # Create a mask using the defined blue range
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # Apply morphological operations to reduce noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Close small holes
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # Remove small noise
    
    # Find contours in the processed mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by area (adjust min_area as needed)
    min_area = 500
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    
    # Write coordinates to text file
    with open('coordinates.txt', 'w') as file:
        for idx, contour in enumerate(filtered_contours):
            file.write(f"Pool {idx + 1} Coordinates:\n")
            for point in contour:
                x, y = point[0]
                file.write(f"{x},{y}\n")
            file.write("\n")  # Separate pools with a newline
    
    # Draw the contours on the original image (red color in BGR)
    output_img = img.copy()
    cv2.drawContours(output_img, filtered_contours, -1, (0, 0, 255), 1)
    
    # Save the output image
    cv2.imwrite('output_image.jpg', output_img)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python pool_detector.py <input_image.jpg>")
        sys.exit(1)
    detect_pools(sys.argv[1])
