from array import array
from pathlib import Path
import struct
import tempfile
import unittest
import zlib
from projects.examples.demo import png


class PngTests(unittest.TestCase):
    def test_rejects_short_buffer_before_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'invalid.png'
            with self.assertRaises(ValueError):
                png(path, array('f', b'\x01' * 16), 4, 4)
            self.assertFalse(path.exists())

    def test_byte_mask_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'mask.png'
            png(path, array('f', list(bytes([0, 1, 1, 0]))), 2, 2)
            data = path.read_bytes()
            offset, compressed = 8, b''
            while offset < len(data):
                size = struct.unpack('!I', data[offset:offset+4])[0]
                kind = data[offset+4:offset+8]
                body = data[offset+8:offset+8+size]
                if kind == b'IDAT':
                    compressed += body
                offset += size + 12
            self.assertEqual(zlib.decompress(compressed), b'\0\0\xff\0\xff\0')
