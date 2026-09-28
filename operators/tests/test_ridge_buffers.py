import ctypes
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch
import numpy as np
import operators.ridge_ops as ridge
class RidgeBufferTests(unittest.TestCase):
    def test_all_small_borders_and_views(self):
        rng=np.random.default_rng(80)
        for h in range(1,10):
            for w in range(1,10):
                a=rng.random((h,w),dtype=np.float32)
                for view in (a,a.T,a[::-1]):
                    expected=ridge.ridge_response_numpy(view)
                    result=np.empty(view.shape,np.float32)
                    self.assertIs(ridge.ridge_response(view,'native',out=result),result)
                    np.testing.assert_array_equal(result,expected)

    def test_invalid_values_never_modify_output(self):
        for backend in ['native','numpy']:
            for value in [np.nan,np.inf,-np.inf,-.001,1.001]:
                gray=np.full((9,13),.5,np.float32);gray[-1,-1]=value
                output=np.full(gray.shape,17,np.float32)
                with self.assertRaises(ValueError):ridge.ridge_response(gray,backend,out=output)
                np.testing.assert_array_equal(output,17)

    def test_output_contract_and_partial_overlap(self):
        gray=np.zeros((9,13),np.float32)
        readonly=np.empty_like(gray);readonly.flags.writeable=False
        unaligned=np.ndarray(gray.shape,np.float32,buffer=bytearray(gray.nbytes+1),offset=1)
        for output in [gray,np.zeros((2,3),np.float32),np.zeros(gray.shape,np.float64),
                       np.zeros((13,9),np.float32).T,readonly,unaligned]:
            with self.assertRaises(ValueError):ridge.ridge_response(gray,'native',out=output)
        data=np.zeros(118,np.float32)
        with self.assertRaises(ValueError):ridge.ridge_response(data[:-1].reshape(9,13),'native',out=data[1:].reshape(9,13))

    def test_native_abi_rejects_partial_overlap(self):
        ridge.ridge_response(np.zeros((1,1),np.float32),'native')
        data=np.linspace(0,1,118,dtype=np.float32);before=data.copy()
        p=ctypes.POINTER(ctypes.c_float)
        fn=ridge._NATIVE.metal_ridge
        for source,target in [(data[:-1],data[1:]),(data[1:],data[:-1])]:
            self.assertEqual(fn(source.ctypes.data_as(p),target.ctypes.data_as(p),13,9),1)
            np.testing.assert_array_equal(data,before)

    def test_unaligned_and_readonly_input_supported(self):
        gray=np.ndarray((9,13),np.float32,buffer=bytearray(9*13*4+1),offset=1)
        gray[:]=np.random.default_rng(3).random(gray.shape)
        expected=ridge.ridge_response_numpy(gray);gray.flags.writeable=False
        np.testing.assert_array_equal(ridge.ridge_response(gray,'native'),expected)

    def test_numpy_and_auto_fallback_output_reuse(self):
        gray=np.random.default_rng(4).random((17,19),dtype=np.float32);out=np.empty_like(gray)
        expected=ridge.ridge_response_numpy(gray)
        self.assertIs(ridge.ridge_response(gray,'numpy',out=out),out)
        np.testing.assert_array_equal(out,expected)
        with patch.object(ridge,'_NATIVE',None),patch.object(ridge.Path,'exists',return_value=False):
            self.assertIs(ridge.ridge_response(gray,'auto',out=out),out)
            np.testing.assert_array_equal(out,expected)

    def test_independent_buffers_on_multiple_threads(self):
        ridge.ridge_response(np.zeros((1,1),np.float32),'native')
        def run(seed):
            gray=np.random.default_rng(seed).random((51,73),dtype=np.float32);out=np.empty_like(gray)
            expected=ridge.ridge_response_numpy(gray)
            for _ in range(5):
                ridge.ridge_response(gray,'native',out=out)
                np.testing.assert_array_equal(out,expected)
            return True
        with ThreadPoolExecutor(max_workers=4) as pool:self.assertTrue(all(pool.map(run,range(8))))


if __name__=='__main__':unittest.main()
