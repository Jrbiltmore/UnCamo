# Script for anomaly detection between IR and visible images
# Anomaly Detection Module for Evolved Epithelial Modulation Detection

import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim

class AnomalyDetector:
    def __init__(self):
        pass

    def compute_difference(self, image1, image2):
        '''
        Compute the absolute difference between two images.
        
        Args:
            image1 (np.array): First input image.
            image2 (np.array): Second input image.
        
        Returns:
            np.array: Absolute difference image.
        '''
        if image1.shape != image2.shape:
            raise ValueError("Input images must have the same dimensions")
        difference = cv2.absdiff(image1, image2)
        return difference

    def structural_similarity(self, image1, image2):
        '''
        Compute the Structural Similarity Index (SSIM) between two images.
        
        Args:
            image1 (np.array): First input image.
            image2 (np.array): Second input image.
        
        Returns:
            float: SSIM score between the two images.
            np.array: SSIM image.
        '''
        if image1.shape != image2.shape:
            raise ValueError("Input images must have the same dimensions")
        ssim_score, ssim_image = ssim(image1, image2, full=True)
        return ssim_score, (ssim_image * 255).astype(np.uint8)

    def threshold_difference(self, difference_image, threshold_value=50):
        '''
        Apply a threshold to the difference image to isolate anomalies.
        
        Args:
            difference_image (np.array): Input difference image.
            threshold_value (int): Threshold value to segment the difference.
        
        Returns:
            np.array: Thresholded binary image.
        '''
        _, thresholded_image = cv2.threshold(difference_image, threshold_value, 255, cv2.THRESH_BINARY)
        return thresholded_image

    def highlight_anomalies(self, image, anomaly_map, color=(0, 0, 255)):
        '''
        Highlight anomalies on the original image based on the anomaly map.
        
        Args:
            image (np.array): Input image where anomalies are to be highlighted.
            anomaly_map (np.array): Binary map of detected anomalies.
            color (tuple): Color to use for highlighting anomalies (default is red).
        
        Returns:
            np.array: Image with anomalies highlighted.
        '''
        if len(anomaly_map.shape) == 2:  # If anomaly map is grayscale
            anomaly_map_colored = cv2.cvtColor(anomaly_map, cv2.COLOR_GRAY2BGR)
        else:
            anomaly_map_colored = anomaly_map
        
        highlighted_image = cv2.addWeighted(image, 0.7, anomaly_map_colored, 0.3, 0)
        highlighted_image[anomaly_map > 0] = color
        return highlighted_image

    def detect_anomalies(self, image1, image2, threshold_value=50):
        '''
        Perform end-to-end anomaly detection between two images.
        
        Args:
            image1 (np.array): First input image (e.g., from IR camera).
            image2 (np.array): Second input image (e.g., from visible camera).
            threshold_value (int): Threshold for detecting anomalies.
        
        Returns:
            np.array: Anomaly map highlighting differences.
        '''
        # Step 1: Compute the absolute difference between the two images
        difference_image = self.compute_difference(image1, image2)
        
        # Step 2: Apply a threshold to the difference to isolate anomalies
        anomaly_map = self.threshold_difference(difference_image, threshold_value)
        
        return anomaly_map

    def detect_ssim_anomalies(self, image1, image2):
        '''
        Detect anomalies using Structural Similarity Index (SSIM).
        
        Args:
            image1 (np.array): First input image (e.g., from IR camera).
            image2 (np.array): Second input image (e.g., from visible camera).
        
        Returns:
            np.array: Map of SSIM-based anomalies.
        '''
        _, ssim_image = self.structural_similarity(image1, image2)
        
        # Anomalies are where SSIM is low (dark regions)
        anomaly_map = cv2.threshold(255 - ssim_image, 50, 255, cv2.THRESH_BINARY)[1]
        
        return anomaly_map

    def detect_anomalies_with_contours(self, image1, image2, threshold_value=50):
        '''
        Detect anomalies between two images and highlight them with contours.
        
        Args:
            image1 (np.array): First input image (e.g., from IR camera).
            image2 (np.array): Second input image (e.g., from visible camera).
            threshold_value (int): Threshold for detecting anomalies.
        
        Returns:
            np.array: Image with contours around detected anomalies.
        '''
        # Step 1: Detect anomalies using difference
        anomaly_map = self.detect_anomalies(image1, image2, threshold_value)
        
        # Step 2: Find contours of the anomalies
        contours, _ = cv2.findContours(anomaly_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Step 3: Draw contours on the original image
        output_image = image1.copy()
        cv2.drawContours(output_image, contours, -1, (0, 255, 0), 2)
        
        return output_image
