# Bubble Sheet Processing and Recognition System

## Introduction
This module automates the recognition and grading of bubble sheet answers. By preprocessing the scanned sheets and leveraging OCR, SVM classifiers, and pretrained models, the system accurately detects filled bubbles, maps them to answers, and exports the results into a structured format.

---

## Features
- **Automatic Alignment**: Corrects skewed or rotated bubble sheets.
- **Bubble Detection**: Identifies and processes filled bubbles using advanced contour detection.
- **Answer Recognition**:
  - **OCR**: Detects text and symbols for answers.
- **Grading**: Maps detected answers to the answer key for automatic scoring.
- **Output Export**: Saves the recognized answers and scores in an Excel sheet.

---

## Installation

### Prerequisites
Ensure the following are installed:
- Python 3.8+
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
- Required Python libraries:
  ```bash
  pip install numpy opencv-python pytesseract scikit-learn torchvision torch matplotlib
  ```

---

## Usage

### 1. Preprocessing and Alignment
Align the scanned bubble sheets

### 2. Bubble Detection
Detect and extract filled bubbles

### 3. Answer Recognition
Recognize the answers in detected bubbles:
- **OCR**

### 4. Grading
Grade the answers against the provided answer key

### 5. Export Results
Save the scores and recognized answers to an Excel file for review.
---

## Test Cases
The module was tested on:
- Bubble sheets with varied layouts (e.g., number of columns, bubble sizes).
- Sheets with partially filled, overfilled, or crossed bubbles.
- Multiple lighting conditions and noise levels.

---

## References
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
- [OpenCV Documentation](https://opencv.org/)
- [LeNet-5](http://yann.lecun.com/exdb/lenet/)
- [Scikit-learn](https://scikit-learn.org/)
- [PyTorch Documentation](https://pytorch.org/)

---

## Additional Notes
- For bubble sheets with highly ambiguous markings, preprocessing (e.g., denoising and contour filtering) is critical.
- Pretrained DNN models are optional but improve performance for complex inputs.
- Ensure proper calibration of bubble sheet templates for consistent alignment.
```

This README is tailored specifically for the bubble sheet module. Let me know if you want additional refinements!
