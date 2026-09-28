# 三维球面鲁棒测量

`fit_sphere(points, threshold=..., min_radius=..., max_radius=..., min_inliers=..., min_inlier_ratio=..., ...)` 从已知三维点拟合球心和半径。4 点 RANSAC 抑制离群点；有足够支持后，用全部当前内点做代数初值和几何径向残差最小化，再重新归类内点。**返回球心、半径上的径向残差、RMS、内点掩码会最后重算**，不是上一次迭代的旧结果。

```python
from operators.sphere_ops import fit_sphere

result = fit_sphere(cloud_points_m, threshold=.005,
                    min_radius=.02, max_radius=.5,
                    min_inliers=200, min_inlier_ratio=.5)
if result['success']:
    center_m, radius_m = result['center'], result['radius']
```

输入点、阈值、半径同单位，可为米或毫米。RANSAC 的 `max_iterations` 是上限，在当前支持达到要求后按四点采样概率与 `confidence` 收缩预算。共面或近乎共面的支持集默认因第三奇异值比例过小而失败；只有一小片可见球冠时也可能无法可靠定位球心，这比编造一个大半径球更合理。`min_spread_ratio` 可由用户明确调节。一个随机离群点恰好落在半径容差内时会被算作几何内点，这是径向阈值的自然边界；接口不提供语义判断、扫描仪标定或协方差。

4 项专项测试覆盖无噪声球、半球、带离群点球面、平面/线退化、错误半径范围、最终残差独立重算和非法输入。合成 1,000 球面点、200 离群点、3 mm 径向噪声：种子 827 的球心误差 **0.118 mm**、半径误差 **0.301 mm**，1,000 个真表面点全保留、零误接纳。四个固定种子中，一例有 3 个随机点碰巧落入 15 mm 半径门限，均在报告中保留。21 次完整 API 短基准：1,200 点、约 17% 离群中位 **2.24 ms**；10,000 点、20% 离群中位 **11.14 ms**。原始数据 `projects/output/continuous_batch5/sphere_benchmark/benchmark.json`。这些是合成误差，不代表实际球面扫描精度。
