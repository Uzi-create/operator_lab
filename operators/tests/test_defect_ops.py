import unittest
import cv2
import numpy as np
from operators.defect_ops import local_defect_contrast,group_defect_fragments


class LocalContrastTest(unittest.TestCase):
    def test_against_scalar_reference(self):
        rng=np.random.default_rng(81)
        for shape in [(1,1),(1,9),(9,1),(11,13)]:
            image=rng.random(shape,dtype=np.float32);mask=rng.random(shape)>.3
            result=local_defect_contrast(image,mask,1,3,.02,1,.01)
            for y in range(shape[0]):
                for x in range(shape[1]):
                    values=[];capacity=0
                    for yy in range(max(0,y-3),min(shape[0],y+4)):
                        for xx in range(max(0,x-3),min(shape[1],x+4)):
                            if max(abs(yy-y),abs(xx-x))<=1:continue
                            capacity+=1
                            if mask[yy,xx]:values.append(float(image[yy,xx]))
                    self.assertEqual(result['support_count'][y,x],len(values))
                    self.assertEqual(result['geometric_count'][y,x],capacity)
                    expected_valid=bool(mask[y,x] and values and len(values)>=capacity*.01)
                    self.assertEqual(result['valid'][y,x],expected_valid)
                    if expected_valid:
                        residual=float(image[y,x])-np.mean(values)
                        score=abs(residual)/np.sqrt(np.var(values)+.02**2)
                        self.assertAlmostEqual(float(result['score'][y,x]),score,delta=2e-5)
                    else:self.assertEqual(result['score'][y,x],0)

    def test_masked_background_does_not_bias(self):
        image=np.full((60,60),.5,np.float32);mask=np.zeros((60,60),np.uint8);mask[10:50,10:50]=1
        image[mask==0]=1
        result=local_defect_contrast(image,mask)
        self.assertLess(float(result['score'].max()),1e-5)
        image[30,30]=.8;result=local_defect_contrast(image,mask)
        self.assertAlmostEqual(float(result['score'][30,30]),15,places=4)

    def test_bright_dark_symmetry_and_empty_support(self):
        image=np.full((40,40),.5,np.float32);image[20,20]=.8;mask=np.ones_like(image,np.uint8)
        a=local_defect_contrast(image,mask);b=local_defect_contrast(1-image,mask)
        np.testing.assert_allclose(a['score'],b['score'],atol=1e-5)
        out=local_defect_contrast(image,np.zeros_like(mask))
        self.assertFalse(out['valid'].any());self.assertTrue(np.isfinite(out['score']).all())
        self.assertEqual(float(out['score'].max()),0)

    def test_invalid_parameters(self):
        image=np.ones((5,5),np.float32);mask=np.ones_like(image,np.uint8)
        for kwargs in [dict(inner_radius=3,outer_radius=3),dict(outer_radius=1.5),dict(noise_floor=0),
                       dict(noise_floor=1e-300),dict(min_support_fraction=2),dict(min_support=0)]:
            with self.assertRaises(ValueError):local_defect_contrast(image,mask,**kwargs)
        with self.assertRaises(ValueError):local_defect_contrast(image,np.full((5,5),np.nan))


class GroupingTest(unittest.TestCase):
    def test_aligned_fragments_merge_without_changing_pixels(self):
        mask=np.zeros((60,100),np.uint8);mask[20,10:31]=1;mask[20,36:61]=1
        before=mask.copy();groups=group_defect_fragments(mask,max_gap=6)
        self.assertEqual(len(groups),1);self.assertEqual(groups[0]['fragment_count'],2)
        self.assertEqual(groups[0]['area_px'],46);np.testing.assert_array_equal(mask,before)
        self.assertEqual(groups[0]['bbox_xywh'],[10,20,51,1])

    def test_parallel_and_perpendicular_are_separate(self):
        mask=np.zeros((80,100),np.uint8);mask[20,10:31]=1;mask[24,10:31]=1
        self.assertEqual(len(group_defect_fragments(mask)),2)
        mask[:]=0;mask[20,10:31]=1;mask[24:45,35]=1
        self.assertEqual(len(group_defect_fragments(mask)),2)

    def test_gap_boundary_and_transitive_chain(self):
        mask=np.zeros((50,100),np.uint8)
        for a,b in [(10,21),(26,37),(42,53)]:mask[20,a:b]=1
        self.assertEqual(len(group_defect_fragments(mask,max_gap=6)),1)
        self.assertEqual(len(group_defect_fragments(mask,max_gap=5.9)),3)

    def test_rotation_and_small_components(self):
        mask=np.zeros((80,80),np.uint8)
        cv2.line(mask,(10,10),(25,25),1,1);cv2.line(mask,(29,29),(45,45),1,1)
        self.assertEqual(len(group_defect_fragments(mask,max_gap=6)),1)
        mask[:]=0;mask[20,20]=1
        self.assertEqual(group_defect_fragments(mask),[])
        self.assertEqual(len(group_defect_fragments(mask,min_area=1)),1)

    def test_component_limit_and_invalid(self):
        mask=np.zeros((30,30),np.uint8);mask[::3,::3]=1
        with self.assertRaises(ValueError):group_defect_fragments(mask,min_area=1,max_components=10)
        for kwargs in [dict(max_gap=-1),dict(max_angle_deg=91),dict(max_lateral=float('nan')),dict(min_area=0)]:
            with self.assertRaises(ValueError):group_defect_fragments(mask,**kwargs)
        self.assertEqual(group_defect_fragments(np.zeros((1,1),np.uint8)),[])

if __name__=='__main__':unittest.main()
