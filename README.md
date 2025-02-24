# Swimming Pool Detection in Aerial Images

This project detects swimming pools in aerial images and generates:
1. A text file (`coordinates.txt`) containing the boundary coordinates of the detected pools.
2. Cropped images of the detected pools, saved in the `cropped_pools` folder (if using Version 2 code).
3. An output image (`output_image_with_outlines.jpg`) with the detected pools outlined in red (if using version 1 code).

---

## **Setup Instructions**

### **1. Prerequisites**
- Python 3.x
- OpenCV (`opencv-python`)
- NumPy (`numpy`)

### **2. Install Dependencies**
Run the following command to install the required libraries:

```bash
pip install opencv-python numpy
```
---
## **Approach**
  - **Image Loading and Conversion:**
      Load the input image and convert it from the BGR color space to the HSV color space, which is more effective for color-based segmentation.
  - **Color Thresholding:**
      Define a range for blue colors in the HSV space to create a mask that isolates potential pool regions.
  - **Morphological Operations:**
      Clean the mask using morphological operations to remove noise and fill gaps in the detected regions.
  - **Contour Detection:**
      Identify contours in the processed mask to detect closed regions that could be pools.
  - **Contour Filtering:**
      Filter contours based on area to exclude small regions that are unlikely to be pools.
  - **Output Generation:**
      Save the coordinates of the detected pools to a text file and draw red outlines around the pools on the original image.
