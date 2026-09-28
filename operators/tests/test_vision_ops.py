import unittest
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np
import operators.vision_ops as v
from operators.fast_stats import compute


class LocalStatsTests(unittest.TestCase):
    def test_native_against_scalar_and_fallback(self):
        rng=np.random.default_rng(97)
        for shape in [(1,1),(1,9),(8,1),(13,17)]:
            for dtype in [np.float32,np.float64]:
                gray=rng.random(shape).astype(dtype);mask=rng.random(shape)>.25
                for inner,outer in [(-1,0),(-1,3),(1,4),(-1,4096)]:
                    kw=dict(inner=inner,outer=outer,min_support_fraction=.1)
                    a=compute(gray,mask,backend='native',**kw);b=compute(gray,mask,backend='opencv',**kw)
                    for key in a:np.testing.assert_allclose(a[key],b[key],atol=2e-5,rtol=2e-5)
                    for y,x in np.ndindex(shape):
                        values=[];capacity=0
                        for yy in range(max(0,y-outer),min(shape[0],y+outer+1)):
                            for xx in range(max(0,x-outer),min(shape[1],x+outer+1)):
                                if inner>=0 and max(abs(yy-y),abs(xx-x))<=inner:continue
                                capacity+=1
                                if mask[yy,xx]:values.append(float(gray[yy,xx]))
                        self.assertEqual(a['support_count'][y,x],len(values))
                        self.assertEqual(a['geometric_count'][y,x],capacity)
                        if a['valid'][y,x]:
                            self.assertAlmostEqual(a['background'][y,x],np.mean(values),places=6)
                            self.assertAlmostEqual(a['variance'][y,x],np.var(values),places=6)

    def test_workspace_resize_concurrency_and_views(self):
        def work(seed):
            rng=np.random.default_rng(seed)
            for shape in [(70,90),(3,2),(99,77)]:
                gray=rng.random(shape,dtype=np.float32).T
                a=compute(gray,backend='native');b=compute(gray,backend='opencv')
                np.testing.assert_allclose(a['score'],b['score'],atol=2e-5)
            return True
        with ThreadPoolExecutor(max_workers=4) as pool:self.assertTrue(all(pool.map(work,range(4))))

    def test_dynamic_and_background_mask(self):
        gray=np.full((51,61),.4,np.float32);mask=np.zeros(gray.shape,bool);mask[8:-8,8:-8]=True
        gray[~mask]=1;gray[25,30]=.8
        out=v.dynamic_threshold(gray,mask,radius=4,offset=.2,polarity='bright')
        self.assertEqual(np.count_nonzero(out),1);self.assertTrue(out[25,30])
        corrected=v.illumination_correct(np.full((20,30),.7,np.float32),radius=4)
        np.testing.assert_allclose(corrected['image'],.5,atol=1e-6)


class RegionTests(unittest.TestCase):
    def test_hysteresis_excludes_isolated_weak_and_mask(self):
        a=np.zeros((20,30),np.float32);a[5,3:15]=.4;a[5,3]=1;a[15,3:15]=.4
        self.assertEqual(v.hysteresis_threshold(a,.3,.8).sum(),12)
        mask=np.ones(a.shape,bool);mask[5,7]=False
        self.assertEqual(v.hysteresis_threshold(a,.3,.8,mask).sum(),4)

    def test_holes_boundary_and_area(self):
        a=np.ones((20,30),bool);a[4:7,4:7]=False;a[:5,20:22]=False
        self.assertEqual(v.fill_holes(a).sum()-a.sum(),9)
        np.testing.assert_array_equal(v.fill_holes(a,max_area=8),a)
        self.assertFalse(v.fill_holes(np.zeros((2,2),bool)).any())

    def test_regions(self):
        a=np.zeros((40,50),np.uint8);a[3:8,4:14]=1;a[30,30]=1
        self.assertEqual(v.select_regions(a,min_area=2).sum(),50)
        r=v.region_features(a,min_area=2)[0]
        self.assertEqual(r['bbox_xywh'],[4,3,10,5]);self.assertEqual(r['area'],50)
        np.testing.assert_allclose(r['centroid_xy'],[8.5,5]);self.assertAlmostEqual(r['orientation_deg'],0)


class MeasurementTests(unittest.TestCase):
    def test_matching_multiple_and_roi_coordinates(self):
        rng=np.random.default_rng(8);template=rng.random((11,13),dtype=np.float32)
        image=np.zeros((90,100),np.float32);image[20:31,15:28]=template;image[60:71,65:78]=template
        matches=v.match_template(image,template,min_score=.99)
        self.assertEqual({tuple(m['bbox_xywh']) for m in matches},{(15,20,13,11),(65,60,13,11)})
        self.assertEqual(v.match_template(image,template,min_score=.99,search_box=(50,50,40,30))[0]['bbox_xywh'],[65,60,13,11])
        with self.assertRaises(ValueError):v.match_template(image,np.ones((3,3),np.float32))

    def test_caliper_subpixel_and_reversal(self):
        x=np.arange(120,dtype=np.float32);edge=55.3
        gray=np.tile((.5+.4*np.tanh((x-edge)/1.5)).astype(np.float32),(30,1))
        result=v.measure_edges(gray,(5,15),(110,15),polarity='bright',threshold=.02)
        self.assertEqual(len(result['edges']),1);self.assertAlmostEqual(result['edges'][0]['xy'][0],edge,delta=.15)
        reverse=v.measure_edges(gray,(110,15),(5,15),polarity='dark',threshold=.02)
        self.assertAlmostEqual(reverse['edges'][0]['xy'][0],edge,delta=.15)
        with self.assertRaises(ValueError):v.measure_edges(gray,(0,0),(110,0),width=9)

    def test_robust_line_with_outliers(self):
        rng=np.random.default_rng(1);x=np.linspace(-20,20,100)
        points=np.column_stack((x,2*x+3+rng.normal(0,.05,len(x))))
        points=np.vstack((points,rng.uniform(-40,40,(30,2))))
        result=v.fit_line(points,threshold=.2,min_inliers=80)
        self.assertGreater(result['inliers'].sum(),95)
        self.assertAlmostEqual(result['direction'][1]/result['direction'][0],2,delta=.01)
        self.assertLess(result['rms'],.08)

    def test_robust_circle_and_degenerate(self):
        rng=np.random.default_rng(2);angle=np.linspace(0,2*np.pi,100,endpoint=False)
        points=np.column_stack((10+20*np.cos(angle),-5+20*np.sin(angle)))+rng.normal(0,.05,(100,2))
        points=np.vstack((points,rng.uniform(-40,40,(30,2))))
        result=v.fit_circle(points,threshold=.3,min_inliers=80)
        np.testing.assert_allclose(result['center'],[10,-5],atol=.1);self.assertAlmostEqual(result['radius'],20,delta=.1)
        self.assertGreater(result['angular_coverage_deg'],350)
        with self.assertRaises(ValueError):v.fit_circle(np.column_stack((np.arange(10),np.arange(10))))
        with self.assertRaises(ValueError):v.fit_line(np.ones((5,2)))

    def test_invalid_parameters(self):
        a=np.zeros((10,10),np.float32)
        with self.assertRaises(ValueError):v.dynamic_threshold(a,offset=-1)
        with self.assertRaises(ValueError):v.select_regions(a,min_area=0)
        with self.assertRaises(ValueError):v.hysteresis_threshold(a,2,1)
        with self.assertRaises(ValueError):compute(a,outer=5000)


if __name__=='__main__':unittest.main()
