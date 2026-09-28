import unittest
import numpy as np
from operators.robot_ops import (Intrinsics,depth_to_points,project_points,transform_points,
                       voxel_downsample,fit_plane,plane_heights,estimate_rigid_transform,depth_normals)


class RobotTests(unittest.TestCase):
    def setUp(self):self.camera=Intrinsics(8,6,100,110,3.5,2.5)

    def test_depth_units_and_projection_roundtrip(self):
        depth=np.full((6,8),1200,np.uint16);depth[2,3]=0
        result=depth_to_points(depth,self.camera,depth_scale=.001,stride=2)
        self.assertEqual(len(result['points']),12)
        np.testing.assert_allclose(result['points'][:,2],1.2)
        projected=project_points(result['points'],self.camera)
        np.testing.assert_allclose(projected['pixels'],result['pixels'],atol=1e-12)
        self.assertTrue(projected['inside_image'].all())

    def test_missing_depth_and_empty_cloud(self):
        depth=np.full((6,8),np.nan,np.float64);depth[1,1]=1;depth[0,0]=np.inf
        result=depth_to_points(depth,self.camera)
        self.assertEqual(result['points'].shape,(1,3))
        depth[:]=0;result=depth_to_points(depth,self.camera)
        self.assertEqual(result['points'].shape,(0,3))
        self.assertEqual(voxel_downsample(result['points'],.1)['points'].shape,(0,3))
        self.assertEqual(project_points(result['points'],self.camera)['pixels'].shape,(0,2))

    def test_projection_behind_camera(self):
        result=project_points([[0,0,-1],[0,0,0],[100,0,1]],self.camera)
        self.assertFalse(result['inside_image'].any());self.assertTrue(np.isnan(result['pixels'][:2]).all())

    def test_voxels_centroids_negative_and_mapping(self):
        p=np.array([[.01,0,0],[.09,0,0],[-.01,0,0],[1,0,0]])
        r=voxel_downsample(p,.1,min_points=2)
        np.testing.assert_allclose(r['points'],[[.05,0,0]])
        np.testing.assert_array_equal(r['counts'],[2]);np.testing.assert_array_equal(r['source_to_voxel'],[0,0,-1,-1])

    def test_plane_with_outliers_and_heights(self):
        rng=np.random.default_rng(7);xy=rng.uniform(-1,1,(400,2))
        p=np.column_stack((xy,1+.1*xy[:,0]+rng.normal(0,.0005,len(xy))))
        p=np.vstack((p,rng.uniform(-1,2,(80,3))))
        r=fit_plane(p,threshold=.003,min_inliers=350)
        self.assertGreaterEqual(r['inliers'].sum(),400);self.assertLess(r['rms'],.001)
        self.assertAlmostEqual(abs(r['normal'][2]),1/np.sqrt(1.01),delta=.001)
        self.assertAlmostEqual(plane_heights([[0,0,1]],r['normal'],r['offset'])[0],0,delta=.001)
        with self.assertRaises(ValueError):fit_plane(np.column_stack((np.arange(10),np.zeros((10,2)))))

    def test_transform_and_known_correspondence_alignment(self):
        rng=np.random.default_rng(9);source=rng.random((20,3))
        transform=np.array([[0,-1,0,.3],[1,0,0,-.2],[0,0,1,.5],[0,0,0,1.]])
        target=transform_points(source,transform)
        result=estimate_rigid_transform(source,target)
        np.testing.assert_allclose(result['transform'],transform,atol=1e-12)
        np.testing.assert_allclose(transform_points(target,np.linalg.inv(transform)),source,atol=1e-12)
        with self.assertRaises(ValueError):transform_points(source,np.diag([2,1,1,1]))
        with self.assertRaises(ValueError):estimate_rigid_transform(np.ones((3,3)),np.ones((3,3)))

    def test_depth_normals_and_discontinuities(self):
        depth=np.ones((6,8),np.float32)
        result=depth_normals(depth,self.camera)
        self.assertEqual(result['valid'].sum(),24)
        np.testing.assert_allclose(result['normals'][result['valid']],np.tile([0,0,-1],(24,1)),atol=1e-6)
        depth[:,4:]=2;result=depth_normals(depth,self.camera,max_depth_jump=.02)
        self.assertFalse(result['valid'][:,3:5].any())

    def test_invalid_arguments(self):
        with self.assertRaises(ValueError):Intrinsics(8,6,0,100,4,3)
        with self.assertRaises(ValueError):depth_to_points(np.ones((2,2)),self.camera)
        with self.assertRaises(ValueError):depth_to_points(np.ones((6,8)),self.camera,depth_scale=0)
        with self.assertRaises(ValueError):voxel_downsample(np.ones((2,3)),0)
        with self.assertRaises(ValueError):plane_heights([[0,0,1]],[0,0,0],0)


if __name__=='__main__':unittest.main()
