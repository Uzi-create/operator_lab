# 带目标法向的点到平面 ICP

`icp_point_to_plane(source, target, target_normals, max_distance=..., ...)` 在每次**精确最近邻**查询后，使用对应目标点的法向最小化局部点到平面残差，返回 `T_target_from_source`。`target` 可为原始 Nx3 点，也可为已缓存的 `NearestNeighborIndex`；`target_normals` 必须与目标原始顺序逐一对应，所有法向非零且有限，内部归一化。

```python
from operators.registration_ops import NearestNeighborIndex
from operators.icp_plane_ops import icp_point_to_plane

with NearestNeighborIndex(target_points, backend='native') as index:
    result = icp_point_to_plane(source_points, index, target_normals,
                                max_distance=.05, trim_fraction=.9,
                                min_overlap=.6)
    if result['converged']:
        T_target_from_source = result['transform']
```

距离、平移和阈值共享输入点单位，旋转容差为弧度。`max_distance` 做三维最近邻距离门控；`max_plane_residual` 可进一步限制沿目标法向的残差；`trim_fraction` 保留绝对平面残差最小的一部分。使用 6 维局部线性化、正交刚体旋转和固定对应集的短线搜索。最终返回的目标索引、三维距离、点到平面残差、内点与 RMS 都在**最终矩阵**上重新算。达到迭代上限给 `converged=False`；损失不能降低给 `status='stalled'`。局部收敛并不能证明全局真值。

单一平面无法约束沿平面的平移和绕法向的旋转，雅可比矩阵缺秩时会明确拒绝，不会制造六自由度精度。几何中须有多个独立法向，目标法向方向可正可负，但应来自相同目标点、相同坐标系。初始位姿必须在最近邻门限内；重复结构、法向噪声、遮挡、错误单位仍会影响结果。这个接口不计算目标法向；有组织化深度图可先用 `robot_ops.depth_normals`，同时确保只传有效法向对应的点。

4 项专项测试覆盖三个互相独立表面的真值配准、末态最近邻/残差独立复算、目标索引缓存、NumPy 与原生最近邻后端等价、单平面退化和非法法向。合成 1,560 源点、1,800 目标点，包含 120 个远离群点：旋转误差 **0.00732°**、平移误差 **0.105 mm**，离群零接纳、6 次迭代。21 次短基准共用缓存 C++ KD 树，中位约 **6.23 ms**；原 point-to-point ICP 同场景约 **6.43 ms**。点到平面使用了额外的已知法向，不能把这组时延或 RMS 当作公平的纯算法精度优劣比较。四种子结果和采样在 `projects/output/continuous_batch9/icp_plane_benchmark/benchmark.json`，尚未验证实际扫描仪。
