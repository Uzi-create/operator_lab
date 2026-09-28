# 三维直线与已观测线段

`fit_line_3d(points, threshold=..., min_inliers=..., min_inlier_ratio=..., min_span=...)` 先用两点 RANSAC 寻找三维直线，再对内点做正交最小二乘（TLS），返回线上的一点、单位方向、**已观测内点投影范围**的两端点、跨度、逐点垂直距离与 RMS。距离、端点及 `min_span` 与输入点同单位。

```python
from operators.line3d_ops import fit_line_3d

result = fit_line_3d(seam_points_m, threshold=.005,
                     min_inliers=80, min_span=.1)
if result['success']:
    point, direction = result['point'], result['direction']
    observed_segment = result['segment']
```

方向符号采用“绝对值最大的分量为正”的确定性约定；无额外坐标先验时，直线本身没有前后方向。`segment` 仅覆盖内点实际采到的最远投影，不能当作未见到的整条管道或焊缝长度。`max_iterations` 是 RANSAC 上限，找到足够支持后按两点成功概率与 `confidence` 自适应减少抽样；若支持不足、点云退化或跨度未达要求会明确失败。最终距离、内点、跨度和 RMS 从返回直线重新计算；`success` 也不证明目标是语义上的轨道/焊缝。

4 项专项测试包括完整/旋转直线、噪声离群、最终距离独立重算、非直线/重合/跨度不足和非法输入。合成 1,000 线点、200 离群点、3 mm 横向高斯噪声：方向误差 **0.01535°**，与真值轴线的垂直偏移 **0.0940 mm**，999 个线点保留、零离群误接纳。21 次短基准完整 API 中位：1,200 点约 **0.912 ms**；10,000 点、20% 离群约 **7.72 ms**。原始报告 `projects/output/continuous_batch7/line3d_benchmark/benchmark.json`。仅合成数据，尚无实际点云标定或现场精度验证。
