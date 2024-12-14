import pytesseract
pytesseract.pytesseract.tesseract_cmd = "C:/Program Files/Tesseract-OCR/tesseract.exe"
from cv2 import findContours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, drawContours
from imutils import grab_contours, contours
import skimage.io as io
import matplotlib.pyplot as plt
import numpy as np
from skimage.exposure import histogram
from matplotlib.pyplot import bar
from skimage.color import rgb2gray,rgb2hsv

# Convolution:
from scipy.signal import convolve2d
from scipy import fftpack
import math

from skimage.util import random_noise, invert
from skimage.filters import median
from skimage.feature import canny
from skimage.measure import label
from skimage.color import label2rgb
from skimage.morphology import dilation, square, disk


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

def gamma_correction(image, c, gamma): 
    return c * image ** gamma

def global_thresholding(image):
    hist = np.zeros(256)
    image = (image*255).astype(np.uint8) 
    for pixel in image.flatten():
        hist[pixel] += 1
    total_pixels = np.sum(hist)
    initial_threshold = np.round(np.sum(np.arange(256) * hist) / total_pixels)
    old_threshold = 0
    while initial_threshold != old_threshold:
        old_threshold = initial_threshold
        lower = np.arange(0, int(initial_threshold))
        higher = np.arange(int(initial_threshold), 256)
        lower_mean = np.round(np.sum(lower * hist[lower] / np.sum(hist[lower]))) if np.sum(hist[lower]) != 0 else 0
        higher_mean = np.round(np.sum(higher * hist[higher] / np.sum(hist[higher]))) if np.sum(hist[higher]) != 0 else 0
        initial_threshold = np.round((higher_mean + lower_mean) / 2)
    binary_image = np.where(image < initial_threshold, 0, 255).astype(np.uint8)
    return binary_image
        