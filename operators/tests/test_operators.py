from array import array
import math
import random
import unittest
from operators import run


def reference(src, w, h, radius, adaptive=False, scale=0.08):
    result = []
    for y in range(h):
        for x in range(w):
            values = [src[yy*w+xx]
                      for yy in range(max(0, y-radius), min(h, y+radius+1))
                      for xx in range(max(0, x-radius), min(w, x+radius+1))]
            mean = math.fsum(values)/len(values)
            # Direct centered variance, independent of integral-image formula.
            variance = math.fsum((v-mean)**2 for v in values)/len(values)
            weight = variance/(variance+scale*scale)
            result.append(mean + weight*(src[y*w+x]-mean) if adaptive else mean)
    return result


class OperatorsTest(unittest.TestCase):
    def test_against_independent_reference(self):
        rng = random.Random(42)
        for w, h in [(1, 1), (1, 9), (7, 1), (3, 4), (17, 13)]:
            src = array('f', (rng.random() for _ in range(w*h)))
            for radius in [0, 1, 3, 100]:
                for mode in ['mean', 'mean_naive', 'adaptive']:
                    with self.subTest(w=w, h=h, radius=radius, mode=mode):
                        expected = reference(src, w, h, radius, mode == 'adaptive')
                        actual = run(src, w, h, mode, radius)
                        self.assertLess(max(abs(a-b) for a,b in zip(actual, expected)), 2e-6)

    def test_constant_and_large_radius(self):
        src = array('f', [0.4])*24
        for mode in ['mean', 'mean_naive', 'adaptive']:
            self.assertEqual(run(src, 6, 4, mode, 2147483647), src)

    def test_threshold_equality(self):
        self.assertEqual(list(run(array('f', [0, 0.5, 1]), 3, 1,
                                  'threshold', parameter=0.5)), [0, 0, 1])

    def test_impulse(self):
        src = array('f', [0])*25
        src[12] = 1
        result = run(src, 5, 5, radius=1)
        self.assertAlmostEqual(result[12], 1/9, places=7)
        self.assertEqual(result[0], 0)

    def test_invalid_inputs(self):
        for values in [[float('nan')], [float('inf')], [-0.1], [1.1]]:
            with self.assertRaises(ValueError):
                run(array('f', values), 1, 1)
        for kwargs in [dict(width=0), dict(height=2), dict(radius=-1),
                       dict(radius=1.5), dict(mode='missing'),
                       dict(parameter=float('inf')), dict(mode='adaptive', parameter=0)]:
            args = dict(width=1, height=1)
            args.update(kwargs)
            with self.assertRaises(ValueError):
                run(array('f', [0]), **args)
        with self.assertRaises(TypeError):
            run([0], 1, 1)

    def test_adaptive_scale_controls_smoothing(self):
        src = array('f', [0, 0, 1, 1, 1])
        weak = run(src, 5, 1, 'adaptive', parameter=0.01)
        strong = run(src, 5, 1, 'adaptive', parameter=1)
        self.assertLess(sum(abs(a-b) for a,b in zip(src, weak)),
                        sum(abs(a-b) for a,b in zip(src, strong)))

if __name__ == '__main__':
    unittest.main()
