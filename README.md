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
