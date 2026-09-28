
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import ctypes
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
from projects.metal.inspect_surface import _components,Parameters,polygon_mask
import operators.ridge_ops as v2
class HardeningTests(unittest.TestCase):
    def test_boxes_enclose_scaled_pixels_and_stay_inside(self):
        binary=np.zeros((100,120),np.uint8);binary[30:45,103:120]=255
        response=np.ones(binary.shape,np.float32)
        boxes,_=_components(binary,response,120/173,100/149,'discoloration_candidate',Parameters())
        self.assertEqual(len(boxes),1)
        x,y,w,h=boxes[0]['bbox_xywh']
        self.assertLessEqual(x,103/(120/173));self.assertLessEqual(y,30/(100/149))
        self.assertGreaterEqual(x+w,120/(120/173));self.assertGreaterEqual(y+h,45/(100/149))
        self.assertLessEqual(x+w,173);self.assertLessEqual(y+h,149)

    def test_rounding_roi_edge_is_clipped(self):
        mask=polygon_mask((10,10),[(8.9,8.9),(9.9,8.9),(9.9,9.9)])
        self.assertEqual(mask.shape,(10,10));self.assertEqual(mask[9,9],255)

    def test_unloadable_native_auto_fallback_and_explicit_error(self):
        image=np.full((3,4),.5,np.float32)
        with patch.object(v2,'_NATIVE',None),patch.object(Path,'exists',return_value=True),patch.object(ctypes,'CDLL',side_effect=OSError('wrong architecture')):
            np.testing.assert_array_equal(v2.ridge_response(image,'auto'),v2.ridge_response_numpy(image))
            with self.assertRaisesRegex(RuntimeError,'rebuild'):v2.ridge_response(image,'native')

if __name__=='__main__':unittest.main()
