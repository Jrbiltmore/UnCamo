# Script for image preprocessing (alignment and normalization)
# Advanced Image Preprocessing Module for Evolved Epithelial Modulation Detection

import cv2
import numpy as np
from scipy import ndimage
from skimage import exposure

class ImagePreprocessor:
    def __init__(self):
        pass

    def normalize_image(self, image, target_range=(0, 255)):
        '''
        Normalize an image to a target intensity range.
        
        Args:
            image (np.array): Input image to be normalized.
            target_range (tuple): The range to which the image will be normalized (default is 0-255).
            
        Returns:
            np.array: Normalized image.
        '''
        image_min, image_max = np.min(image), np.max(image)
        normalized_image = (image - image_min) / (image_max - image_min) * (target_range[1] - target_range[0]) + target_range[0]
        return normalized_image.astype(np.uint8)

    def resize_image(self, image, target_size):
        '''
        Resize an image to the target size while preserving aspect ratio.
        
        Args:
            image (np.array): Input image to be resized.
            target_size (tuple): Target size as (width, height).
        
        Returns:
            np.array: Resized image.
        '''
        h, w = image.shape[:2]
        scale = min(target_size[0] / w, target_size[1] / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return resized_image

    def sharpen_image(self, image):
        '''
        Apply sharpening to the input image using a custom kernel.
        
        Args:
            image (np.array): Input image to be sharpened.
        
        Returns:
            np.array: Sharpened image.
        '''
        kernel = np.array([[0, -1, 0], 
                           [-1, 5,-1], 
                           [0, -1, 0]])
        sharpened_image = cv2.filter2D(image, -1, kernel)
        return sharpened_image

    def equalize_histogram(self, image):
        '''
        Apply histogram equalization to improve the contrast of the image.
        
        Args:
            image (np.array): Input grayscale image.
        
        Returns:
            np.array: Image with equalized histogram.
        '''
        if len(image.shape) == 2:
            # For grayscale images
            equalized_image = cv2.equalizeHist(image)
        else:
            # For color images, apply histogram equalization to each channel
            channels = cv2.split(image)
            eq_channels = [cv2.equalizeHist(ch) for ch in channels]
            equalized_image = cv2.merge(eq_channels)
        return equalized_image

    def apply_gaussian_blur(self, image, kernel_size=(5, 5)):
        '''
        Apply Gaussian blur to smooth the image.
        
        Args:
            image (np.array): Input image.
            kernel_size (tuple): Size of the Gaussian kernel.
        
        Returns:
            np.array: Blurred image.
        '''
        blurred_image = cv2.GaussianBlur(image, kernel_size, 0)
        return blurred_image

    def rotate_image(self, image, angle):
        '''
        Rotate the image by a specified angle.
        
        Args:
            image (np.array): Input image.
            angle (float): The angle by which to rotate the image.
        
        Returns:
            np.array: Rotated image.
        '''
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h))
        return rotated_image

    def adaptive_thresholding(self, image, max_value=255, block_size=11, C=2):
        '''
        Apply adaptive thresholding to binarize the image.
        
        Args:
            image (np.array): Input grayscale image.
            max_value (int): Maximum value to use with THRESH_BINARY.
            block_size (int): Size of the pixel neighborhood that is used to calculate the threshold.
            C (int): Constant subtracted from the mean or weighted mean.
        
        Returns:
            np.array: Binarized image using adaptive thresholding.
        '''
        if len(image.shape) != 2:
            raise ValueError("Adaptive thresholding requires a grayscale image")
        thresholded_image = cv2.adaptiveThreshold(image, max_value, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                  cv2.THRESH_BINARY, block_size, C)
        return thresholded_image

    def apply_clahe(self, image, clip_limit=2.0, tile_grid_size=(8, 8)):
        '''
        Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) to enhance local contrast.
        
        Args:
            image (np.array): Input grayscale image.
            clip_limit (float): Threshold for contrast limiting.
            tile_grid_size (tuple): Size of the grid for histogram equalization.
        
        Returns:
            np.array: Image with enhanced local contrast using CLAHE.
        '''
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        return clahe.apply(image)
