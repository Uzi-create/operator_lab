
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dataclasses import replace
import unittest
import cv2
import numpy as np
from projects.metal.inspect_surface import inspect as v1,Parameters
from projects.metal.inspect_surface_v2 import inspect,ridge_response,ridge_response_numpy,Refinement


class RefinementTests(unittest.TestCase):
    def test_ridge_rejects_steps_and_ramps(self):
        for row in [np.linspace(0,1,80,dtype=np.float32),np.array([0]*40+[1]*40,dtype=np.float32)]:
            gray=np.tile(row,(60,1))
            self.assertEqual(float(ridge_response(gray).max()),0.)

    def test_dark_bright_and_inversion(self):
        gray=np.full((80,100),.5,np.float32)
        gray[20:60,30]=.1;gray[20:60,70]=.9
        result=ridge_response(gray)
        self.assertGreater(result[40,30],.39);self.assertGreater(result[40,70],.39)
        np.testing.assert_allclose(result,ridge_response(1-gray),atol=1e-7)

    def test_native_matches_numpy(self):
        rng=np.random.default_rng(91)
        for shape in [(1,1),(1,19),(17,1),(3,4),(91,73)]:
            gray=rng.random(shape,dtype=np.float32)
            for sample in [gray,gray.T]:
                np.testing.assert_allclose(ridge_response(sample,'native'),ridge_response_numpy(sample),atol=1e-7)

    def test_whole_component_filter_never_adds_internal_pixels(self):
        image=np.full((240,320,3),140,np.uint8);roi=np.full(image.shape[:2],255,np.uint8)
        cv2.line(image,(60,100),(240,100),(60,60,60),3)
        r1,m1=v1(image,roi);r2,m2=inspect(image,roi)
        self.assertGreater(len(r2['scratch_candidates']),0)
        self.assertLessEqual(len(r2['scratch_candidates']),len(r1['scratch_candidates']))
        self.assertFalse(np.any((m2['scratch_mask']>0)&(m1['scratch_mask']==0)))
        np.testing.assert_array_equal(m2['rejected_v1_pixels']|m2['scratch_mask'],m1['scratch_mask'])

    def test_border_is_review_only(self):
        image=np.full((240,320,3),140,np.uint8);roi=np.full(image.shape[:2],255,np.uint8)
        cv2.line(image,(60,10),(230,10),(60,60,60),3)
        report,maps=inspect(image,roi)
        self.assertEqual(report['scratch_candidates'],[])
        self.assertGreater(len(report['border_review_candidates']),0)
        self.assertTrue(all(c['kind']=='border_review_candidate' for c in report['border_review_candidates']))
        self.assertFalse(np.any((maps['border_review_mask']>0)&(maps['inspection_mask']>0)))

    def test_refinement_validation(self):
        image=np.full((100,100,3),140,np.uint8);roi=np.ones((100,100),np.uint8)
        for refine in [Refinement(ridge_floor=float('nan')),Refinement(min_supported_fraction=1.1),Refinement(border_margin=2)]:
            with self.assertRaises(ValueError):inspect(image,roi,refinement=refine)
        with self.assertRaises(ValueError):ridge_response(np.zeros((2,2),np.float32),'bad')
        with self.assertRaises(ValueError):ridge_response(np.array([[np.nan]],np.float32))

if __name__=='__main__':unittest.main()
