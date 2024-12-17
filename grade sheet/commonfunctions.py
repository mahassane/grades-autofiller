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

alpha = 1.0
beta = 0.0
# Function to perform the perspective transformation
def perspective_transform(img,binary):

    if img is None:
        print(f"Error: Unable to load the image.")
    else:
        # Convert the image to grayscale
        binary_image=0
        if binary==0:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Apply thresholding to create a binary image
            _, binary_image = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
        else :
            binary_image=img
        # cf.show_images([binary_image])
        # Find contours in the binary image
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Find the largest contour based on area
        print(len(contours))
        largest_contour = max(contours, key=cv2.contourArea)
        # Approximate the contour to a polygon
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx_polygon = cv2.approxPolyDP(largest_contour, epsilon, True)

        # Get the four corners of the polygon
        corners = approx_polygon.reshape(-1, 2)
        corners = sorted(corners, key=lambda x: x[1])
        # Separate the sorted corners into top and bottom
        top_corners = sorted(corners[:2], key=lambda x: x[0])
        bottom_corners = sorted(corners[2:], key=lambda x: x[0])

        # Concatenate the sorted corners
        sorted_corners = np.concatenate([bottom_corners, top_corners])

        # Define the destination points for the perspective transformation
        dst_points = np.float32([[0, img.shape[0]], [img.shape[1], img.shape[0]], [0, 0], [img.shape[1], 0]])

        # Calculate the perspective transformation matrix
        matrix = cv2.getPerspectiveTransform(sorted_corners.astype(np.float32), dst_points)

        # Apply the perspective transformation to the image
        warped_img = cv2.warpPerspective(img, matrix, (img.shape[1], img.shape[0]))
        return warped_img
    
def invert_image(img):
    clone =img
    adjusted_img = cv2.addWeighted(img, alpha, np.zeros(img.shape, img.dtype), 0, beta)
    # cf.show_images([adjusted_img])
    img=perspective_transform(adjusted_img,0)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh_image = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,91, 5)
    thresh_image=255-thresh_image
    trial = perspective_transform(thresh_image,1)
    # cf.show_images([thresh_image])
    transform=perspective_transform(thresh_image,1)
    return transform


''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
arr=[[] for _ in range (6)]
def readImage(imgpath):
    img = cv2.imread(imgpath)
    if img is None:
        print(f"Error: Unable to read image at {imgpath}")
        return None
    transformed_img = perspective_transform(img, 0)
    inverted_img = invert_image(transformed_img)
    show_images([inverted_img])
    return inverted_img

def getVerticalLines(inverted_img):
    # Check for invalid input
    if inverted_img is None:
        print("Error: Input image is None.")
        return []
    
    # Edge detection
    edges = cv2.Canny(inverted_img, 50, 150, apertureSize=3)
    cv2.imshow("Edges Detected", edges)  # Debugging step: visualize edges
    cv2.waitKey(0)  # Press a key to continue
    
    # Line detection
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=260,
                            minLineLength=160, maxLineGap=20)
    
    # Handle the case where no lines are detected
    if lines is None:
        print("Error: No lines detected.")
        return []
    
    # Sort lines based on x-coordinate
    sorted_lines = sorted(lines, key=lambda line: line[0][0])
    print("Detected and sorted vertical lines:", sorted_lines)
    
    # Filter lines to avoid duplicates
    filtered_lines = [sorted_lines[0]]
    for line in sorted_lines[1:]:
        prev_x = filtered_lines[-1][0][0]
        cur_x = line[0][0]
        if cur_x - prev_x >= 20:
            filtered_lines.append(line)
    
    return filtered_lines


def getBlocks(lines,segments,inverted_img):
    x1=lines[0][0][0]
    idx=0
    for line in lines:
        x2=line[0][0]
        if(x2-x1>segments[0]):
            block = inverted_img[0:4000, x1:x2]
            break      
        idx=idx+1
    y1 = 150
    while y1 < block.shape[0]:
        cell = block[y1:y1 + 200, :]
        show_images([cell])
        arr[segments[1]].append(cell)
        y1 += 220
    return [block,idx]


    
