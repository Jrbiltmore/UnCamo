
# Unit Tests for Polarization Analysis Functions

import unittest
import cv2
import numpy as np
from src.polarization_analysis import PolarizationAnalyzer

class TestPolarizationAnalysis(unittest.TestCase):
    
    def setUp(self):
        # Initialize the PolarizationAnalyzer instance
        self.analyzer = PolarizationAnalyzer()
        # Create a sample polarized image (grayscale gradient)
        self.image = np.linspace(0, 255, 256, dtype=np.uint8).reshape(16, 16)

    def test_calculate_polarization(self):
        # Test the calculation of the degree of polarization
        polarization_map = self.analyzer.calculate_polarization(self.image, angle_step=5)
        self.assertIsNotNone(polarization_map)
        self.assertEqual(polarization_map.shape, self.image.shape)
    
    def test_analyze_degree_of_polarization(self):
        # Test the analysis of polarization based on a threshold
        high_polarization_map = self.analyzer.analyze_degree_of_polarization(self.image, threshold=50)
        self.assertIsNotNone(high_polarization_map)
        self.assertEqual(high_polarization_map.shape, self.image.shape)

    def test_detect_polarization_anomalies(self):
        # Create a reference image with no modulation
        reference_image = self.image.copy()
        reference_image[8:12, 8:12] = 128  # Simulate modulation in the test image

        # Test the detection of polarization anomalies
        anomaly_map = self.analyzer.detect_polarization_anomalies(self.image, reference_image, threshold=50)
        self.assertIsNotNone(anomaly_map)
        self.assertEqual(anomaly_map.shape, self.image.shape)

    def test_visualize_polarization(self):
        # Test visualization of the polarization map
        polarization_map = self.analyzer.calculate_polarization(self.image)
        try:
            self.analyzer.visualize_polarization(polarization_map)
        except Exception as e:
            self.fail(f"Visualization raised an exception: {e}")

    def test_compare_polarization_across_channels(self):
        # Create a color image with different polarization for each channel
        color_image = np.dstack([self.image, self.image, self.image])
        polarization_maps = self.analyzer.compare_polarization_across_channels(color_image)

        # Test that each channel's polarization map is returned
        self.assertIn('R', polarization_maps)
        self.assertIn('G', polarization_maps)
        self.assertIn('B', polarization_maps)

    def test_highlight_polarization_anomalies(self):
        # Create a binary anomaly map for testing
        anomaly_map = np.zeros_like(self.image)
        anomaly_map[8:12, 8:12] = 255

        # Test highlighting of polarization anomalies
        highlighted_image = self.analyzer.highlight_polarization_anomalies(self.image, anomaly_map, color=(255, 0, 0))
        self.assertIsNotNone(highlighted_image)
        self.assertEqual(highlighted_image.shape, self.image.shape)
