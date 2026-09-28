# 重力对齐的三维定向包围盒

`gravity_aligned_box(points, up=..., min_footprint=...)` 把三维点投影到垂直于 `up` 的平面，用 OpenCV C++ 的 `minAreaRect` 求最小面积水平外接矩形；然后沿该方向和重力方向用 float64 **重新计算所有输入点的极值**，确保包围所有传入点。返回 `center`、三个边长 `size`、列向量组成的右手系 `axes`、8 个角点、水平面积与体积。第一轴取较长的水平边，第三轴即 `up`。

```python
from operators.box_ops import gravity_aligned_box

result = gravity_aligned_box(cluster_points_world, up=(0, 0, 1))
if result['success']:
    center, size, corners = result['center'], result['size'], result['corners']
```

`up` 不必是 Z，也不必事先归一化。点和 `min_footprint` 使用相同单位，通常为米。少于 3 点或水平投影近乎共线返回 `success=False`，不会编造宽度。近乎正方形时 `orientation_ambiguous=True`：包围盒位置/尺寸仍可用，但水平朝向不唯一。算法不自动删除离群点；一个离群点即可扩张整个包围盒。输入若是深度相机只见到的物体表面，输出只包围**可见点**，高度不能当作完整物体厚度或碰撞体积。水平最小面积朝向由重新居中的 float32 投影求出，之后以 float64 重新包围；若坐标跨度极大，应核对设备精度。

4 项专项测试覆盖已知旋转长方体、任意重力方向、所有点包围、表面高度为零、离群扩张、正方形歧义及非法/退化输入。带全部 8 个极值顶点的合成长方体，0.8×0.35×0.4 m 尺寸误差约 4.3e-9 / 9.9e-9 / 0 m。31 次短采样的完整 API 中位：2,008 点由 0.799 降至 0.708 ms；100,008 点由 26.96 降至 23.02 ms，优化是把三个方向的最终投影合并为一次矩阵乘法。优化前后原始报告分别在 `projects/output/continuous_batch4/box_benchmark` 和 `box_benchmark_optimized`。
