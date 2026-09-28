from array import array
from dataclasses import replace
import math
import random
import statistics
import unittest
from operators.competition import grasp_depth,TargetEvidenceGate


def estimate(values,**kwargs):
    return grasp_depth(array('f',values),bytes([1])*len(values),len(values),1,
                       fx=200,fy=200,cx=0,cy=0,min_samples=3,**kwargs)


class GraspDepthTest(unittest.TestCase):
    def test_near_outliers_do_not_capture_surface(self):
        result=estimate([.18]*25+[.30]*70+[.9]*5)
        self.assertTrue(result.accepted)
        self.assertAlmostEqual(result.depth_m,.3,places=6)
        self.assertEqual(result.cluster_count,70)
        self.assertAlmostEqual(result.runner_ratio,25/70)
        self.assertEqual(sum(result.inliers),70)
        self.assertEqual(result.pixel,(59.5,0))

    def test_ambiguity_rejection(self):
        result=estimate([.3]*50+[.5]*50,min_cluster_fraction=.45)
        self.assertFalse(result.accepted)
        self.assertEqual(result.reason,'ambiguous_surfaces')
        self.assertIsNone(result.point_m)
        self.assertEqual(sum(result.inliers),0)

    def test_holes_do_not_borrow_outside_depth(self):
        depth=array('f',[.3]*100)
        for i in range(20): depth[i]=math.nan if i%2 else 0
        mask=bytes([1]*20+[0]*80)
        result=grasp_depth(depth,mask,10,10,fx=200,fy=200,cx=5,cy=5,min_samples=3)
        self.assertFalse(result.accepted)
        self.assertEqual(result.valid_count,0)
        self.assertEqual(result.area,20)
        result=estimate([.3]*10+[0]*40)
        self.assertFalse(result.accepted)
        self.assertEqual(result.reason,'insufficient_support')

    def test_depth_units_and_backprojection(self):
        result=grasp_depth(array('f',[.4]*12),bytes([1])*12,4,3,
                           fx=100,fy=200,cx=1.5,cy=1,min_samples=3)
        self.assertTrue(result.accepted)
        self.assertAlmostEqual(result.point_m[0],0)
        self.assertAlmostEqual(result.point_m[1],0)
        self.assertAlmostEqual(result.size_m[0],.016,places=6)
        self.assertFalse(estimate([300]*20).accepted)  # millimeters are NOT silently accepted

    def test_densest_band_against_brute_force(self):
        rng=random.Random(31)
        for _ in range(30):
            values=array('f',(rng.uniform(.1,.8) for _ in range(50)))
            band=.04
            expected=max(sum(0<=b-a<=band for b in values) for a in values)
            result=estimate(values,band_width=band,min_cluster_fraction=.01,
                            max_mad=1,ambiguity_ratio=1)
            self.assertEqual(result.cluster_count,expected)
            if result.accepted:
                inliers=[v for v,k in zip(values,result.inliers) if k]
                self.assertAlmostEqual(result.depth_m,statistics.median(inliers))
                self.assertLessEqual(max(inliers)-min(inliers),band)

    def test_mad_and_configuration(self):
        result=estimate([.29,.30,.31]*10,max_mad=.001)
        self.assertEqual(result.reason,'depth_spread')
        for kwargs in [dict(band_width=0),dict(min_cluster_fraction=0),dict(max_mad=-1),
                       dict(ambiguity_ratio=1.1),dict(min_depth=2,max_depth=1)]:
            with self.assertRaises(ValueError): estimate([.3]*20,**kwargs)
        result=grasp_depth(array('f',[.3]),b'\0',1,1,fx=1,fy=1,cx=0,cy=0)
        self.assertFalse(result.accepted)


class EvidenceGateTest(unittest.TestCase):
    def setUp(self):
        self.good=estimate([.3]*20)
        self.gate=TargetEvidenceGate()

    def update(self,t,estimate=None,key='product-A',now=None):
        return self.gate.update(self.good if estimate is None else estimate,
                                stamp=t,now=t if now is None else now,target_key=key)

    def test_confirmation_and_time_span(self):
        self.assertFalse(self.update(0).ready)
        self.assertFalse(self.update(.01).ready)
        self.assertFalse(self.update(.02).ready)
        self.assertTrue(self.update(.08).ready)

    def test_duplicate_stale_and_missing_never_ready(self):
        for t in [0,.04,.08]: self.update(t)
        result=self.update(.08,now=.09)
        self.assertFalse(result.ready);self.assertFalse(result.fresh)
        self.assertEqual(result.state,'out_of_order')
        result=self.update(.1,now=.3)
        self.assertEqual(result.state,'stale_frame');self.assertFalse(result.ready)
        result=self.gate.update(None,stamp=.31,now=.31,target_key='product-A')
        self.assertFalse(result.ready);self.assertFalse(result.fresh)

    def test_target_switch_jump_and_reacquire(self):
        self.update(0)
        result=self.update(.04,key='product-B')
        self.assertEqual(result.state,'different_target')
        jumping=replace(self.good,point_m=(1,1,1))
        result=self.update(.08,jumping)
        self.assertEqual(result.state,'position_jump')
        result=self.update(.5,key='product-B')
        self.assertEqual(result.state,'confirming');self.assertEqual(result.confirmations,1)

    def test_gap_and_lost_reset_confirmation(self):
        for t in [0,.04,.08]: self.update(t)
        result=self.update(.3)
        self.assertEqual(result.confirmations,1);self.assertFalse(result.ready)
        result=self.gate.update(None,stamp=.8,now=.8,target_key='product-A')
        self.assertEqual(result.state,'lost');self.assertIsNone(result.point_m)

    def test_clock_reset_and_future(self):
        self.update(1)
        with self.assertRaises(ValueError): self.update(.9)
        self.assertEqual(self.update(.95).confirmations,1)
        result=self.update(2,now=1)
        self.assertEqual(result.state,'future_frame');self.assertFalse(result.ready)
        self.gate.reset()
        self.assertEqual(self.update(0).confirmations,1)

if __name__=='__main__':unittest.main()
