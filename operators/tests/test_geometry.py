from array import array
from collections import deque
import math
import random
import unittest
from operators import canny, distance_transform, signed_distance, feather


def brute_distance(src,w,h,signed=False):
    foreground=[i for i,v in enumerate(src) if v>0.5]
    background=[i for i,v in enumerate(src) if v<=0.5]
    result=[]
    for i,v in enumerate(src):
        targets=(background if v>0.5 else foreground) if signed else background
        d=min((math.hypot(i%w-j%w,i//w-j//w) for j in targets),default=math.inf)
        result.append(-d if signed and v<=0.5 else d)
    return result


def canny_reference(src,w,h,low,high):
    # Slow direct 25-sample convolution, then scalar gradient/NMS and component labeling.
    k=[1,4,6,4,1]
    blur=[]
    for y in range(h):
        for x in range(w):
            blur.append(math.fsum(k[dy+2]*k[dx+2]*src[min(h-1,max(0,y+dy))*w+min(w-1,max(0,x+dx))]
                                 for dy in range(-2,3) for dx in range(-2,3))/256)
    magnitude=[0.]*(w*h)
    direction=[0]*(w*h)
    sx=[[-1,0,1],[-2,0,2],[-1,0,1]]
    sy=[[-1,-2,-1],[0,0,0],[1,2,1]]
    for y in range(1,h-1):
        for x in range(1,w-1):
            gx=sum(blur[(y+dy)*w+x+dx]*sx[dy+1][dx+1]
                   for dy in range(-1,2) for dx in range(-1,2))/4
            gy=sum(blur[(y+dy)*w+x+dx]*sy[dy+1][dx+1]
                   for dy in range(-1,2) for dx in range(-1,2))/4
            magnitude[y*w+x]=math.hypot(gx,gy)
            angle=math.degrees(math.atan2(gy,gx))%180
            direction[y*w+x]=0 if angle<=22.5 or angle>=157.5 else (1 if angle<67.5 else (2 if angle<=112.5 else 3))
    kept=[0.]*(w*h)
    offsets=[(-1,1),(-w-1,w+1),(-w,w),(-w+1,w-1)]
    for y in range(1,h-1):
        for x in range(1,w-1):
            i=y*w+x
            a,b=offsets[direction[i]]
            if magnitude[i]>magnitude[i+a] and magnitude[i]>=magnitude[i+b]: kept[i]=magnitude[i]
    # Find every connected weak component; keep the entire component only if it contains a strong seed.
    seen=set()
    result=[0.]*(w*h)
    for seed,value in enumerate(kept):
        if value<low or seed in seen: continue
        queue=deque([seed]); seen.add(seed); component=[]
        while queue:
            i=queue.popleft(); component.append(i)
            for yy in range(max(0,i//w-1),min(h,i//w+2)):
                for xx in range(max(0,i%w-1),min(w,i%w+2)):
                    j=yy*w+xx
                    if j not in seen and kept[j]>=low: seen.add(j); queue.append(j)
        if any(kept[i]>=high for i in component):
            for i in component: result[i]=1.
    return result


class GeometryTest(unittest.TestCase):
    def assert_distances(self,actual,expected):
        for a,b in zip(actual,expected):
            if math.isinf(b): self.assertEqual(a,b)
            else: self.assertAlmostEqual(a,b,delta=2e-5)

    def test_distance_bruteforce(self):
        rng=random.Random(95)
        for w,h in [(1,1),(1,11),(13,1),(7,9),(12,8)]:
            for _ in range(4):
                src=array('f',(rng.random() for _ in range(w*h)))
                with self.subTest(w=w,h=h):
                    self.assert_distances(distance_transform(src,w,h),brute_distance(src,w,h))
                    self.assert_distances(signed_distance(src,w,h),brute_distance(src,w,h,True))

    def test_pythagoras_and_threshold(self):
        src=array('f',[1])*35
        src[0]=0.5
        out=distance_transform(src,7,5)
        self.assertEqual(out[4*7+3],5.)
        self.assertEqual(out[0],0.)
        self.assertAlmostEqual(out[8],math.sqrt(2),places=6)

    def test_no_sites_and_no_outside_padding(self):
        for value in [0,1]:
            src=array('f',[value])*15
            expected=math.inf if value else 0
            self.assertEqual(list(distance_transform(src,5,3)),[expected]*15)
            self.assertEqual(list(signed_distance(src,5,3)),[math.inf if value else -math.inf]*15)
            self.assertEqual(list(feather(src,5,3)),[value]*15)

    def test_feather_formula_and_complement(self):
        src=array('f',[0]*8+[1]*8)
        signed=brute_distance(src,16,1,True)
        for width in [0.1,1,3,20]:
            out=feather(src,16,1,width)
            expected=[]
            for d in signed:
                t=max(0,min(1,0.5+d/(2*width)))
                expected.append(t*t*(3-2*t))
            self.assert_distances(out,expected)
            self.assertEqual(sorted(out),list(out))
            complement=feather(array('f',(1-v for v in src)),16,1,width)
            for a,b in zip(out,complement): self.assertAlmostEqual(a+b,1,places=6)

    def test_canny_scalar_reference(self):
        rng=random.Random(25)
        for w,h in [(1,1),(2,7),(9,11),(15,13)]:
            src=array('f',(rng.random() for _ in range(w*h)))
            for low,high in [(0.03125,0.125),(0.0625,0.25),(0.125,0.125)]:
                with self.subTest(w=w,h=h,low=low,high=high):
                    self.assertEqual(list(canny(src,w,h,low,high)),canny_reference(src,w,h,low,high))

    def test_canny_step_and_constant(self):
        for v in [0,0.5,1]:
            self.assertEqual(list(canny(array('f',[v])*400,20,20)),[0]*400)
        src=array('f',(float(x>=10) for y in range(20) for x in range(20)))
        edges=canny(src,20,20)
        self.assertEqual(sum(edges),18)
        self.assertTrue(all(edges[y*20+9]==1 for y in range(1,19)))
        self.assertEqual(sum(canny(src,20,20,1,2)),0)

    def test_canny_hysteresis_connected_weak_edge(self):
        w,h=40,40
        src=array('f',(0 if x<20 else max(0.25,1-y*0.025) for y in range(h) for x in range(w)))
        linked=canny(src,w,h,0.0625,0.4)
        strong_only=canny(src,w,h,0.4,0.4)
        self.assertGreater(sum(linked[y*w+x] for y in range(20,38) for x in (19,20)),10)
        self.assertEqual(sum(strong_only[y*w+x] for y in range(20,38) for x in (19,20)),0)
        isolated=array('f',(0 if x<20 else 0.25 for y in range(h) for x in range(w)))
        self.assertEqual(sum(canny(isolated,w,h,0.0625,0.4)),0)

    def test_geometry_validation(self):
        src=array('f',[0])
        for fn in [canny,distance_transform,signed_distance,feather]:
            for v in [math.nan,math.inf,-1,2]:
                with self.assertRaises(ValueError): fn(array('f',[v]),1,1)
            with self.assertRaises(ValueError): fn(src,2,1)
            with self.assertRaises(TypeError): fn([0],1,1)
        for low,high in [(0,.2),(.3,.1),(.1,math.inf),(-1,1)]:
            with self.assertRaises(ValueError): canny(src,1,1,low,high)
        for width in [0,-1,math.nan]:
            with self.assertRaises(ValueError): feather(src,1,1,width)

if __name__=='__main__': unittest.main()
