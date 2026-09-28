# 重力约束的地面候选与障碍高度

`segment_ground_obstacles(points, up=..., expected_ground_height=..., ...)` 从 N×3 点云中拟合一个与给定向上方向夹角不大于 `max_tilt_deg` 的平面，并按相对高度划分地面、障碍、下方及未分类点。输入坐标和所有距离阈值使用同一单位，通常为米。`up` 可为任意非零三维向量；物体坐标系 Z 不必向上。

```python
from operators.ground_ops import segment_ground_obstacles

result = segment_ground_obstacles(
    points_world, up=(0, 0, 1),
    expected_ground_height=0., height_tolerance=.1,
    max_tilt_deg=10., distance_threshold=.015,
    obstacle_min_height=.05, obstacle_max_height=1.5,
)
if result['success']:
    obstacle_points = points_world[result['obstacles']]
```

RANSAC 从三点生成候选，立即拒绝法向不合重力方向的墙面；可用 `expected_ground_height` 与 `height_tolerance` 限制该平面在坐标原点沿 `up` 方向的截距。点数最多的合格候选经正交最小二乘精修，返回位姿的内点、RMS 和分类再次计算。`max_iterations` 是上限；当当前内点比例足以满足最小支持时，按 `confidence` 和三点采样概率缩短迭代。`ground` 按点到平面的垂直距离判定；`obstacles`、`below` 与 `vertical_heights` 按沿 `up` 的高度判定。平面法向朝向 `up`，返回 `normal·point + offset = 0`。

若不提供预期高度，较大的水平桌面可能在地面较小时获选。专项测试用 1,100 个地面点和 2,300 个桌面点证明：无高度先验时桌面在 0.6 m 获选；给出 `expected_ground_height=0` 后地面获选。`success` 仅说明满足几何支持条件，不代表通行安全、可抓取性或语义“地板”。点云须已配准到明确的坐标系；缺失区域、台阶、坡度、动态物体和重力方向误差仍需现场验证。

4 项专项测试覆盖墙面点数占优、倾斜地面/任意重力方向、桌面竞争、上下方分类、最终平面残差独立重算及失败输入。合成 10,000 点例子有 6,000 墙点、3,000 地面点：地面召回 100%，墙面误入地面 0%，法向误差 0.00450°，截距误差 0.0415 mm。4 个种子的合成检查通过。21 次短基准全 API 中位耗时：墙面占优的 10,000 点 **14.26 ms**（254 次抽样），地面占优的 4,700 点 **2.26 ms**；场景、支持比和迭代数都会改变耗时。原始报告在 `projects/output/continuous_batch3/ground_benchmark/benchmark.json`。
