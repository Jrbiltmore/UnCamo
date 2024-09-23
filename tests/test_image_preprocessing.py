
# Unit Tests for Image Preprocessing Functions

import unittest
import cv2
import numpy as np
from src.image_preprocessing import ImagePreprocessor

class TestImagePreprocessing(unittest.TestCase):
    
    def setUp(self):
        # Initialize the ImagePreprocessor instance
        self.preprocessor = ImagePreprocessor()
        # Create a sample image (grayscale gradient)
        self.image = np.linspace(0, 255, 256, dtype=np.uint8).reshape(16, 16)

    def test_normalize_image(self):
        # Test normalization of the image to 0-255 range
        normalized_image = self.preprocessor.normalize_image(self.image, target_range=(0, 255))
        self.assertEqual(np.min(normalized_image), 0)
        self.assertEqual(np.max(normalized_image), 255)
    
    def test_resize_image(self):
        # Test resizing of the image to a target size
        resized_image = self.preprocessor.resize_image(self.image, target_size=(32, 32))
        self.assertEqual(resized_image.shape, (32, 32))

    def test_sharpen_image(self):
        # Test sharpening of the image
        sharpened_image = self.preprocessor.sharpen_image(self.image)
        self.assertIsNotNone(sharpened_image)

if __name__ == '__main__':
    unittest.main()

    def test_equalize_histogram(self):
        # Test histogram equalization on the image
        equalized_image = self.preprocessor.equalize_histogram(self.image)
        self.assertIsNotNone(equalized_image)
        self.assertEqual(equalized_image.shape, self.image.shape)
    
    def test_apply_gaussian_blur(self):
        # Test Gaussian blur application on the image
        blurred_image = self.preprocessor.apply_gaussian_blur(self.image)
        self.assertIsNotNone(blurred_image)
        self.assertEqual(blurred_image.shape, self.image.shape)

    def test_rotate_image(self):
        # Test image rotation by 90 degrees
        rotated_image = self.preprocessor.rotate_image(self.image, 90)
        self.assertIsNotNone(rotated_image)
        self.assertEqual(rotated_image.shape, self.image.shape)

    def test_adaptive_thresholding(self):
        # Test adaptive thresholding on the image
        thresholded_image = self.preprocessor.adaptive_thresholding(self.image, max_value=255)
        self.assertIsNotNone(thresholded_image)
        self.assertEqual(thresholded_image.shape, self.image.shape)

    def test_apply_clahe(self):
        # Test CLAHE on the image
        clahe_image = self.preprocessor.apply_clahe(self.image)
        self.assertIsNotNone(clahe_image)
        self.assertEqual(clahe_image.shape, self.image.shape)
