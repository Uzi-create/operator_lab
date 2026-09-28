
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dataclasses import replace
import unittest
import cv2
import numpy as np
from projects.metal.inspect_surface import inspect,polygon_mask,Parameters


class MetalTests(unittest.TestCase):
    def scene(self):
        image=np.full((240,320,3),140,np.uint8)
        roi=polygon_mask(image.shape,[(10,10),(309,10),(309,229),(10,229)])
        return image,roi

    def test_constant_surface_has_no_candidates(self):
        image,roi=self.scene();report,maps=inspect(image,roi)
        self.assertEqual(report['scratch_candidates'],[])
        self.assertEqual(report['discoloration_candidates'],[])
        self.assertIsNone(report['oxidation_confirmed'])

    def test_bright_and_dark_scratches(self):
        image,roi=self.scene()
        cv2.line(image,(70,70),(140,110),(20,20,20),3)
        cv2.line(image,(170,160),(250,160),(235,235,235),3)
        report,maps=inspect(image,roi)
        self.assertGreaterEqual(len(report['scratch_candidates']),2)
        for x,y in [(105,90),(210,160)]:
            self.assertTrue(any(a<=x<a+w and b<=y<b+h for a,b,w,h in
                                (r['bbox_xywh'] for r in report['scratch_candidates'])))

    def test_paper_line_and_roi_border_excluded(self):
        image,roi=self.scene()
        roi[:]=0;roi[60:180,60:260]=255
        cv2.line(image,(10,20),(300,20),(0,0,0),3)
        cv2.line(image,(60,80),(60,150),(0,0,0),3)
        report,_=inspect(image,roi)
        self.assertEqual(report['scratch_candidates'],[])

    def test_color_patch_with_good_reference(self):
        image,roi=self.scene()
        image[80:150,180:250]=(60,95,150)
        reference=np.zeros(roi.shape,np.uint8);reference[80:150,65:130]=255
        report,maps=inspect(image,roi,reference_mask=reference)
        self.assertGreater(len(report['discoloration_candidates']),0)
        self.assertEqual(report['color_baseline_source'],'user_supplied_good_patch')
        self.assertIsNone(report['oxidation_confirmed'])
        self.assertEqual(maps['color_mask'][110,210],255)

    def test_neutral_illumination_not_color_oxidation(self):
        image,roi=self.scene()
        yy,xx=np.indices(roi.shape)
        gray=(85+130*np.exp(-((xx-160)**2+(yy-120)**2)/(2*60**2))).astype(np.uint8)
        image[:]=gray[:,:,None]
        report,_=inspect(image,roi)
        self.assertEqual(report['discoloration_candidates'],[])
        self.assertEqual(report['scratch_candidates'],[])

    def test_original_coordinates_and_inputs_unchanged(self):
        image=np.full((480,640,3),140,np.uint8)
        roi=np.ones((480,640),np.uint8)*255
        cv2.line(image,(150,180),(350,180),(20,20,20),6)
        copy=image.copy();maskcopy=roi.copy()
        report,_=inspect(image,roi,Parameters(max_side=320))
        self.assertEqual(report['working_size'],[320,240])
        self.assertTrue(any(x<=250<x+w and y<=180<y+h for x,y,w,h in
                            (r['bbox_xywh'] for r in report['scratch_candidates'])))
        np.testing.assert_array_equal(copy,image);np.testing.assert_array_equal(maskcopy,roi)

    def test_invalid_inputs_and_empty_roi(self):
        image,roi=self.scene()
        for bad in [image[:,:,0],image.astype(np.float32)]:
            with self.assertRaises(ValueError):inspect(bad,roi)
        with self.assertRaises(ValueError):inspect(image,np.zeros_like(roi))
        with self.assertRaises(ValueError):inspect(image,roi,reference_mask=np.zeros_like(roi))
        with self.assertRaises(ValueError):polygon_mask(image.shape,[(0,0),(999,0),(0,20)])
        for params in [replace(Parameters(),scratch_floor=float('nan')),
                       replace(Parameters(),min_color_area=-1),replace(Parameters(),border_margin=2)]:
            with self.assertRaises(ValueError):inspect(image,roi,params)

if __name__=='__main__':unittest.main()
