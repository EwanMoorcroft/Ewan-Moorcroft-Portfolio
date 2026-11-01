"""Hand-calculated guards against wrong scales and QA masks."""
import unittest
import numpy as np
from resilience.raster import clear_land_mask, temperature_celsius, vegetation_index


class LandsatTests(unittest.TestCase):
    def test_temperature_uses_collection_two_scale_and_masks_fill(self):
        actual = temperature_celsius(np.array([0, 45000, 50000]))
        self.assertTrue(np.isnan(actual[0]))
        np.testing.assert_allclose(actual[1:], [29.6609, 46.751], atol=1e-4)

    def test_each_adverse_qa_bit_and_saturation_exclude_pixels(self):
        qa = np.array([64, 1, 2, 4, 8, 16, 32, 128, 64], dtype='uint16')
        sat = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1], dtype='uint16')
        self.assertEqual(clear_land_mask(qa, sat).tolist(), [True] + [False] * 8)

    def test_ndvi_applies_reflectance_offset_before_ratio(self):
        actual = vegetation_index(np.array([20000, 0]), np.array([10000, 10000]))
        self.assertAlmostEqual(actual[0], 0.6470588235294118, places=6)
        self.assertTrue(np.isnan(actual[1]))


if __name__ == '__main__':
    unittest.main()
