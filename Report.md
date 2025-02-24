# **Project Report: Swimming Pool Detection in Aerial Images**

---

## **1. Introduction**

### **1.1 Project Overview**
This project aims to detect swimming pools in aerial images using image processing techniques. The script identifies pools of various shapes (rectangular, oval, irregular) and generates:
1. A text file (`coordinates.txt`) containing the boundary coordinates of the detected pools.
2. Cropped images of the detected pools, saved in the `cropped_pools` folder.
3. An output image (`output_image_with_outlines.jpg`) with the detected pools outlined in red.

### **1.2 Objectives**
- Develop a robust algorithm to detect swimming pools in aerial images.
- Generate accurate boundary coordinates for the detected pools.
- Provide visual outputs (outlined image and cropped images) for further analysis.

---

## **2. Methodology**

### **2.1 Tools and Technologies**
- **Programming Language**: Python
- **Libraries**:
  - OpenCV (`cv2`) for image processing.
  - NumPy (`numpy`) for numerical operations.
- **Input**: A single aerial image in JPEG or PNG format.
- **Output**:
  - `coordinates.txt`: Boundary coordinates of detected pools.
  - `output_image_with_outlines.jpg`: Input image with red outlines around detected pools.
  - `cropped_pools/`: Folder containing cropped images of detected pools.

### **2.2 Algorithm**
1. **Image Preprocessing**:
   - Convert the input image from the BGR color space to the HSV color space.
   - Apply a color threshold to detect cyan-colored regions (pools).

2. **Noise Reduction**:
   - Use morphological operations (closing and opening) to remove noise and fill gaps in the detected regions.

3. **Contour Detection**:
   - Detect contours in the processed mask.
   - Filter out small regions based on area to exclude false positives.

4. **Output Generation**:
   - Save the boundary coordinates of the detected pools in `coordinates.txt`.
   - Draw red outlines around the detected pools on the input image and save it as `output_image_with_outlines.jpg`.
   - Crop the detected pools and save them as separate images in the `cropped_pools` folder.

---

## **3. Implementation**

### **3.1 Key Steps**
1. **Color Segmentation**:
   - Define the HSV range for cyan colors to create a mask.
   - Example:
     ```python
     lower_cyan = np.array([75, 45, 100])   # Lower bound (H, S, V)
     upper_cyan = np.array([120, 255, 255])  # Upper bound (H, S, V)
     ```

2. **Noise Reduction**:
   - Apply morphological operations to clean the mask:
     ```python
     kernel = np.ones((5, 5), np.uint8)
     mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Close small holes
     mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # Remove small noise
     ```

3. **Contour Detection**:
   - Find contours in the processed mask:
     ```python
     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
     ```

4. **Output Generation**:
   - Save coordinates:
     ```python
     with open('coordinates.txt', 'w') as file:
         for idx, contour in enumerate(filtered_contours):
             file.write(f"Pool {idx + 1} Coordinates:\n")
             for point in contour:
                 x, y = point[0]
                 file.write(f"{x},{y}\n")
             file.write("\n")
     ```
   - Draw outlines and save the output image:
     ```python
     cv2.drawContours(output_img, filtered_contours, -1, (0, 0, 255), 2)
     cv2.imwrite('output_image_with_outlines.jpg', output_img)
     ```

---
## **5. Discussion**

### **5.1 Strengths**
- **Accuracy**: The algorithm accurately detects swimming pools of various shapes and sizes.
- **Flexibility**: The script can be adapted to detect other objects by adjusting the color range.
- **Visual Outputs**: The outlined image and cropped images provide clear visual feedback.

### **5.2 Limitations**
- **Color Dependency**: The algorithm relies on color segmentation, which may fail if the pool color varies significantly.
- **Noise Sensitivity**: The script may detect non-pool regions with similar colors (e.g., rooftops, cars).

### **5.3 Future Improvements**
- **Machine Learning**: Use a machine learning model to improve detection accuracy and reduce false positives.
- **Shape Analysis**: Incorporate shape-based features to distinguish pools from other objects.
- **Adaptive Thresholding**: Use adaptive thresholding to handle varying lighting conditions.

---

## **6. Conclusion**

This project successfully detects swimming pools in aerial images using image processing techniques. The script generates accurate boundary coordinates, visual outlines, and cropped images, making it a useful tool for further analysis. Future improvements can enhance the algorithm's robustness and applicability to a wider range of scenarios.

---

## **7. References**
- OpenCV Documentation: https://docs.opencv.org/
- NumPy Documentation: https://numpy.org/doc/

---

## **8. Appendix**

### **8.1 Code**
The complete code for the project is available in the `main.py` scripts in Version 1 and 2 folders.

### **8.2 Sample Input and Output**
Sample input and output files are included in the assest directory.
