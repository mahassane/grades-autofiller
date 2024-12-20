import skimage.io as io
import matplotlib.pyplot as plt
import numpy as np
from skimage.exposure import histogram
from matplotlib.pyplot import bar
from skimage.color import rgb2gray,rgb2hsv
from skimage.transform import rotate

# Convolution:
from scipy.signal import convolve2d
from scipy import fftpack
import math

from skimage.util import random_noise
from skimage.filters import median
from skimage.feature import canny
from skimage.measure import label
import cv2
from skimage.color import label2rgb
import os
from functools import cmp_to_key
from pathlib import Path
from PIL import Image

# Imports for classifying
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier  # MLP is a NN
from sklearn import svm
import numpy as np
import argparse
import imutils 
import cv2
import os
import random


# Depending on library versions on your system, one of the following imports 
from sklearn.model_selection import train_test_split

# Edges
from skimage.filters import sobel_h, sobel, sobel_v,roberts, prewitt

# Show the figures / plots inside the notebook
def show_images(images,titles=None):
    #This function is used to show image(s) with titles by sending an array of images and an array of associated titles.
    # images[0] will be drawn with the title titles[0] if exists
    # You aren't required to understand this function, use it as-is.
    n_ims = len(images)
    if titles is None: titles = ['(%d)' % i for i in range(1,n_ims + 1)]
    fig = plt.figure()
    n = 1
    for image,title in zip(images,titles):
        a = fig.add_subplot(1,n_ims,n)
        if image.ndim == 2: 
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
        n += 1
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_ims)
    plt.show() 


def showHist(img):
    # An "interface" to matplotlib.axes.Axes.hist() method
    plt.figure()
    imgHist = histogram(img, nbins=256)
    
    bar(imgHist[1].astype(np.uint8), imgHist[0], width=0.8, align='center')

"''''''''''''''''''''''''''''''''''''''''''''''''''''''''"
# Enhanced Perspective Transformation
def align_image_using_perspective(image, is_binary=False):
    if image is None:
        raise ValueError("Error: Image is None or not loaded.")

    if not is_binary:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary_image = cv2.threshold(gray_image, 50, 255, cv2.THRESH_BINARY)
    else:
        binary_image = image

    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    polygon = cv2.approxPolyDP(largest_contour, epsilon, True)

    if len(polygon) != 4:
        raise ValueError("Error: Could not find a quadrilateral.")

    corners = sorted(polygon.reshape(-1, 2), key=lambda x: x[1])
    top_corners = sorted(corners[:2], key=lambda x: x[0])
    bottom_corners = sorted(corners[2:], key=lambda x: x[0])
    sorted_corners = np.concatenate([bottom_corners, top_corners])

    height, width = image.shape[:2]
    destination_points = np.float32([[0, height], [width, height], [0, 0], [width, 0]])
    transform_matrix = cv2.getPerspectiveTransform(sorted_corners.astype(np.float32), destination_points)
    aligned_image = cv2.warpPerspective(image, transform_matrix, (width, height))
    return aligned_image

# Enhanced Image Inversion
def enhance_and_invert_image(image):
    adjusted_image = cv2.addWeighted(image, 1.0, np.zeros(image.shape, image.dtype), 0, 0)
    aligned_image = align_image_using_perspective(adjusted_image, False)
    gray = cv2.cvtColor(aligned_image, cv2.COLOR_BGR2GRAY)
    binary_image = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 91, 5)
    inverted_image = 255 - binary_image
    return align_image_using_perspective(inverted_image, True)

# Load and Preprocess Image
def load_and_preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Error: Unable to read image at {image_path}")
    aligned_image = align_image_using_perspective(image, False)
    inverted_image = enhance_and_invert_image(aligned_image)
    return inverted_image

# Detect Vertical Lines
def detect_vertical_lines(image):
    edges = cv2.Canny(image, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=260, minLineLength=160, maxLineGap=20)
    if lines is None:
        raise ValueError("Error: No lines detected.")
    sorted_lines = sorted(lines, key=lambda line: line[0][0])
    filtered_lines = [sorted_lines[0]]
    for line in sorted_lines[1:]:
        if line[0][0] - filtered_lines[-1][0][0] >= 20:
            filtered_lines.append(line)
    return filtered_lines

# Function to extract individual blocks
arr=[[] for _ in range (6)]
def extract_blocks(lines, segment_info, binary_img):
    start_x = lines[0][0][0]
    line_index = 0
    for line in lines:
        end_x = line[0][0]
        if (end_x - start_x > segment_info[0]):  # Width exceeds threshold
            block = binary_img[0:4000, start_x:end_x]
            break
        line_index += 1

    row_start = 150
    while row_start < block.shape[0]:
        cell = block[row_start:row_start + 200, :]  # Extract individual cells
        show_images([cell])
        arr[segment_info[1]].append(cell)  # Append to segment-specific array
        row_start += 220

    return [block, line_index]
