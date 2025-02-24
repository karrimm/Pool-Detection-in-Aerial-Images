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

### **3. Run the Script**
Run the script from the command line, passing the path to the input image as an argument,(Replace input_image.jpg with the path to your test image.). For example:
```bash
python pool_detector.py input_image.jpg
```

---
## **Approach**
1. **Image Loading and Conversion:** Load the input image and convert it from the BGR color space to the HSV color space, which is more effective for color-based segmentation.
2. **Color Thresholding:** Define a range for blue colors in the HSV space to create a mask that isolates potential pool regions.
3. **Morphological Operations:** Clean the mask using morphological operations to remove noise and fill gaps in the detected regions.
4. **Contour Detection:** Identify contours in the processed mask to detect closed regions that could be pools.
5. **Contour Filtering:** Filter contours based on area to exclude small regions that are unlikely to be pools.
6. **Output Generation:** Save the coordinates of the detected pools to a text file and draw red outlines around the pools on the original image.



---
## **Sample Outputs**

- **V2**
  
![image](https://github.com/user-attachments/assets/2aacadd1-0f66-41ff-bcb1-558eb23b9527)

![image](https://github.com/user-attachments/assets/4c93f9c0-bb25-455f-9bb1-762cf5a1e3ce)


- **V1**
  
![image](https://github.com/user-attachments/assets/ff431cee-8dc1-474b-a9d2-bcbee6eba00b)
