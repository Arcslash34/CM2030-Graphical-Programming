# 🎨 CM2030 Graphical Programming - Image Processing Application

Welcome to the **Image Processing Application**! This project showcases an advanced graphical programming solution, including features like image segmentation, real-time face detection, and creative overlays. Built using **JavaScript**, **ml5.js**, **p5.js** and designed for optimal performance on the **Firefox browser**, this application is a demonstration of modern graphical programming techniques.

---

## 🚀 Features

### **1. Image Segmentation**
- **Per-Channel Segmentation**:
  - Users can control the segmentation intensity for Red, Green, and Blue channels using sliders.
  - Threshold sliders allow highlighting or isolating regions based on color characteristics.

- **Color Space Comparisons**:
  - Segmentation in multiple color spaces: RGB, HSV, and Y'CbCr.
  - Insights into the unique characteristics of each color space:
    - **HSV**: Sensitive to variations in color intensity.
    - **Y'CbCr**: Useful for distinguishing brightness and darkness.

### **2. Extensions**
- **Custom Filters**:
  - Added a Y'CbCr filter and a merged filter for advanced image processing.
- **Real-Time Face Detection and Filters**:
  - Utilizes the `ml5.js` library for detecting facial landmarks.
  - Applies anime and cat-themed overlays to video inputs dynamically.

### **3. Real-Time Video Processing**
- Face detection with facial landmark recognition for accurate filter placement.
- Filters dynamically adjust size and position based on face size for a natural appearance.

---

## ⚙️ Setup Instructions

### **Prerequisites**
- A working installation of **Firefox** browser (recommended for testing).
- Basic knowledge of JavaScript and the `ml5.js` library.

### **Steps to Run the Application**
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Arcslash34/CM2030-Graphical-Programming.git
