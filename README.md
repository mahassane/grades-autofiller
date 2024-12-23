
# Grade Sheet Processing and Recognition System

## Introduction
This project automates the recognition of student IDs, grades, and column-based answers from scanned grade sheets. By utilizing advanced image preprocessing techniques, OCR, SVM, and pretrained DNN models, the system extracts, processes, and outputs the required data efficiently.

---

## Features
- **Automatic Alignment**: Aligns scanned grade sheets using perspective transformation.
- **Cell Extraction**: Identifies and extracts individual cells (IDs, grades, and columns) from the grade sheets.
- **Recognition Models**:
  - **OCR**: Recognizes text using Tesseract OCR.
  - **SVM Classifier**: Classifies handwritten digits based on HOG features.
  - **Pretrained DNN**: Uses the LeNet-5 model for digit classification.
- **Data Output**: Exports extracted data into structured Excel sheets for further processing.

---

## Installation

### Prerequisites
Ensure you have the following installed:
- Python 3.8+
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
- Required Python libraries (install via pip):
  ```bash
  pip install numpy opencv-python pytesseract scikit-learn torchvision torch matplotlib
  ```

---

## Usage

### 1. Preprocessing and Alignment
Align the scanned grade sheets using the `align_image_using_perspective` function:
```python
aligned_image = align_image_using_perspective(image)
```

### 2. Cell Extraction
Extract cells (IDs, grades, and columns) using the `load_extracted_cells` function:
```python
cells = load_extracted_cells(output_dir, num_images, num_rows_per_image)
```

### 3. Recognition
Predict the content of cells using one of the following methods:
- **OCR**:
  ```python
  ids = predict_student_ids(images, method="OCR")
  ```
- **SVM**:
  ```python
  grades = predict_col1_answers(col1_images, method="SVM")
  ```
- **DNN**:
  ```python
  digit = predict_digit_DNN(image_path, trained_dnn_model)
  ```

### 4. Export Results
Results can be saved into an Excel file for further analysis.

---

## Directory Structure
```
Output_Result/
    1/
        Student ID__1.jpg
        col1__1.jpg
        col2__1.jpg
        col3__1.jpg
    2/
        Student ID__1.jpg
        col1__1.jpg
        col2__1.jpg
        col3__1.jpg
```

---

## Test Cases
The system was tested on:
- **Variety**: Grade sheets with varying resolutions, noise, and lighting conditions.
- **Metrics**: Accuracy, recall, and F1-score were calculated to assess the models' performance.
- **Comparison**: The strengths and weaknesses of OCR, SVM, and DNN models were analyzed, highlighting the trade-offs between speed and accuracy.

---

## Contributors
- **[Your Name]**: Preprocessing, OCR implementation
- **[Team Member 2]**: SVM training and testing
- **[Team Member 3]**: DNN implementation and analysis
- **[Team Member 4]**: Integration and debugging
- **[Team Member 5]**: Report and documentation

---

## References
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
- [OpenCV Documentation](https://opencv.org/)
- [LeNet-5](http://yann.lecun.com/exdb/lenet/)
- [Scikit-learn](https://scikit-learn.org/)
- [PyTorch Documentation](https://pytorch.org/)

---

## License
This project is licensed under the MIT License - see the LICENSE file for details.

---

## Additional Notes
- For noisy grade sheets, preprocessing steps like adaptive thresholding and edge detection are crucial.
- The pretrained DNN (LeNet-5) can be further fine-tuned with custom datasets for improved performance.
```

This file is structured to guide anyone using or contributing to your project, with clear sections for features, usage, and structure. Let me know if you want additional customization!
