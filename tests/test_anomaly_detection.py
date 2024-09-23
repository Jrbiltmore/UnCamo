
# Unit Tests for Anomaly Detection Functions

import unittest
import cv2
import numpy as np
from src.anomaly_detection import AnomalyDetector

class TestAnomalyDetection(unittest.TestCase):
    
    def setUp(self):
        # Initialize the AnomalyDetector instance
        self.detector = AnomalyDetector()
        # Create two sample images with slight differences
        self.image1 = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
        self.image2 = self.image1.copy()
        self.image2[32:40, 32:40] = 0  # Add a small anomaly in the second image

    def test_compute_difference(self):
        # Test the computation of absolute difference between two images
        difference = self.detector.compute_difference(self.image1, self.image2)
        self.assertIsNotNone(difference)
        self.assertEqual(difference.shape, self.image1.shape)

    def test_threshold_difference(self):
        # Test thresholding the difference image to isolate anomalies
        difference = self.detector.compute_difference(self.image1, self.image2)
        thresholded = self.detector.threshold_difference(difference, threshold_value=50)
        self.assertIsNotNone(thresholded)
        self.assertEqual(thresholded.shape, difference.shape)

if __name__ == '__main__':
    unittest.main()

    def test_highlight_anomalies(self):
        # Test highlighting anomalies on the original image
        difference = self.detector.compute_difference(self.image1, self.image2)
        thresholded = self.detector.threshold_difference(difference, threshold_value=50)
        highlighted = self.detector.highlight_anomalies(self.image1, thresholded, color=(0, 0, 255))
        self.assertIsNotNone(highlighted)
        self.assertEqual(highlighted.shape, self.image1.shape)

    def test_detect_anomalies(self):
        # Test the end-to-end anomaly detection between two images
        anomaly_map = self.detector.detect_anomalies(self.image1, self.image2, threshold_value=50)
        self.assertIsNotNone(anomaly_map)
        self.assertEqual(anomaly_map.shape, self.image1.shape)

    def test_detect_ssim_anomalies(self):
        # Test SSIM-based anomaly detection
        anomaly_map = self.detector.detect_ssim_anomalies(self.image1, self.image2)
        self.assertIsNotNone(anomaly_map)
        self.assertEqual(anomaly_map.shape, self.image1.shape)

    def test_detect_anomalies_with_contours(self):
        # Test detection of anomalies with contours
        anomaly_contour_image = self.detector.detect_anomalies_with_contours(self.image1, self.image2, threshold_value=50)
        self.assertIsNotNone(anomaly_contour_image)
        self.assertEqual(anomaly_contour_image.shape, self.image1.shape)
