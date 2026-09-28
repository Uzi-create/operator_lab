from array import array
import math
import random
import unittest
from operators import run, guided, clahe
from operators.tests.test_operators import reference


def morph_ref(src, w, h, r, maximum):
    op = max if maximum else min
    return [op(src[yy*w+xx]
               for yy in range(max(0,y-r), min(h,y+r+1))
               for xx in range(max(0,x-r), min(w,x+r+1)))
            for y in range(h) for x in range(w)]


def guided_ref(src, w, h, r, scale):
    mean = reference(src,w,h,r)
    variances = []
    for y in range(h):
        for x in range(w):
            vals = [src[yy*w+xx] for yy in range(max(0,y-r),min(h,y+r+1))
                    for xx in range(max(0,x-r),min(w,x+r+1))]
            variances.append(math.fsum((v-mean[y*w+x])**2 for v in vals)/len(vals))
    a = [v/(v+scale*scale) for v in variances]
    b = [(1-v)*m for v,m in zip(a,mean)]
    ma, mb = reference(a,w,h,r), reference(b,w,h,r)
    return [max(0,min(1,aa*v+bb)) for aa,v,bb in zip(ma,src,mb)]


def clahe_ref(src,w,h,tile,clip):
    # Slow scalar oracle: count bins and sum a weighted CDF per pixel.
    centers_x = [(x+min(w,x+tile)-1)/2 for x in range(0,w,tile)]
    centers_y = [(y+min(h,y+tile)-1)/2 for y in range(0,h,tile)]
    maps = {}
    for ty,y in enumerate(range(0,h,tile)):
        for tx,x in enumerate(range(0,w,tile)):
            values = [int(src[yy*w+xx]*255+0.5)
                      for yy in range(y,min(h,y+tile)) for xx in range(x,min(w,x+tile))]
            limit = max(1,clip*len(values)/256)
            hist = [min(values.count(k),limit) for k in range(256)]
            remainder = (len(values)-sum(hist))/256
            maps[tx,ty] = [sum(hist[:k+1])+(k+1)*remainder for k in range(256)]
            maps[tx,ty] = [v/len(values) for v in maps[tx,ty]]
    def weights(pos,centers):
        if pos <= centers[0]: return [(0,1)]
        if pos >= centers[-1]: return [(len(centers)-1,1)]
        low = max(i for i,c in enumerate(centers) if c <= pos)
        alpha = (pos-centers[low])/(centers[low+1]-centers[low])
        return [(low,1-alpha),(low+1,alpha)]
    return [sum(wx*wy*maps[tx,ty][int(src[y*w+x]*255+0.5)]
                for tx,wx in weights(x,centers_x) for ty,wy in weights(y,centers_y))
            for y in range(h) for x in range(w)]


class AdvancedTest(unittest.TestCase):
    def assert_close(self,actual,expected):
        self.assertEqual(len(actual),len(expected))
        self.assertLess(max(abs(a-b) for a,b in zip(actual,expected)),2e-6)

    def test_guided_reference(self):
        rng=random.Random(7)
        for w,h in [(1,1),(1,8),(9,1),(7,6)]:
            src=array('f',(rng.random() for _ in range(w*h)))
            for r in [0,1,3,2147483647]:
                for scale in [0.001,0.12,10]:
                    with self.subTest(w=w,h=h,r=r,scale=scale):
                        self.assert_close(guided(src,w,h,r,scale),guided_ref(src,w,h,r,scale))

    def test_guided_constant_and_identity(self):
        src=array('f',[0.37])*64
        self.assert_close(guided(src,8,8),src)
        src=array('f',[0,0.2,0.7,1])
        self.assert_close(guided(src,2,2,radius=0),src)

    def test_morphology_reference(self):
        rng=random.Random(9)
        for w,h in [(1,1),(1,19),(17,1),(9,11)]:
            src=array('f',(rng.random() for _ in range(w*h)))
            for r in [0,1,3,100]:
                eroded=morph_ref(src,w,h,r,False)
                dilated=morph_ref(src,w,h,r,True)
                expected={'erode':eroded,'dilate':dilated,
                          'open':morph_ref(eroded,w,h,r,True),
                          'close':morph_ref(dilated,w,h,r,False)}
                for mode,values in expected.items():
                    with self.subTest(w=w,h=h,r=r,mode=mode):
                        self.assertEqual(list(run(src,w,h,mode,r)),values)

    def test_morphology_order_and_idempotence(self):
        rng=random.Random(44)
        src=array('f',(rng.random() for _ in range(143)))
        for mode in ['open','close']:
            out=run(src,13,11,mode,2)
            self.assertEqual(out,run(out,13,11,mode,2))
            self.assertTrue(all(a <= b if mode=='open' else a >= b for a,b in zip(out,src)))

    def test_clahe_reference_partial_tiles(self):
        rng=random.Random(11)
        for w,h in [(1,1),(1,9),(9,1),(13,11)]:
            src=array('f',(rng.random() for _ in range(w*h)))
            for tile in [2,4,8,99]:
                for clip in [1,2,1000]:
                    with self.subTest(w=w,h=h,tile=tile,clip=clip):
                        self.assert_close(clahe(src,w,h,tile,clip),clahe_ref(src,w,h,tile,clip))

    def test_clahe_known_cdf(self):
        src=array('f',[0,1])
        self.assert_close(clahe(src,2,1,2,1),[0.5,1])
        src=array('f',[0])*16
        # clip=1, count=16 => clipped mass 1, redistributed mass 15.
        self.assert_close(clahe(src,4,4,4,1),[(1+15/256)/16]*16)
        src=array('f',[i/255 for i in range(256)])
        out=clahe(src,256,1,256,2)
        self.assert_close(out,[(i+1)/256 for i in range(256)])

    def test_advanced_validation(self):
        src=array('f',[0])
        for mode in ['guided','clahe','erode','dilate','open','close']:
            with self.assertRaises(ValueError):
                run(array('f',[float('nan')]),1,1,mode)
        for scale in [0,-1,float('inf')]:
            with self.assertRaises(ValueError): guided(src,1,1,scale=scale)
        for tile,clip in [(0,2),(1,2),(2,0.5),(2,float('nan'))]:
            with self.assertRaises(ValueError): clahe(src,1,1,tile,clip)

if __name__=='__main__':
    unittest.main()
