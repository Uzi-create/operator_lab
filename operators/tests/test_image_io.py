from pathlib import Path
import tempfile
import unittest
import numpy as np
from operators.image_io import read_image, write_image


class ImageIOTests(unittest.TestCase):
    def test_unicode_path_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / '中文划痕.png'
            image = np.full((7, 9, 3), 127, np.uint8)
            write_image(path, image)
            np.testing.assert_array_equal(read_image(path), image)

    def test_failed_write_is_not_silent(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(OSError):
                write_image(Path(directory) / 'missing' / 'image.png',
                            np.zeros((3, 3, 3), np.uint8))
