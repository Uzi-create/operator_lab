# 标定像素射线与平面交点

`camera_rays(pixels, camera, distortion=...)` 把 `(u,v)` 像素坐标转换为相机光学坐标系的单位射线。`intersect_image_plane(..., plane_normal, plane_offset, T_frame_from_camera=...)` 用这些射线与已知平面 `normal·point + offset = 0` 求交。若不提供 `T_frame_from_camera`，平面就在相机坐标系；否则平面与输出点都在目标 frame。相机光学坐标仍为 X 向右、Y 向下、Z 向前。

```python
from operators.ray_plane_ops import intersect_image_plane

# 已知工作台在 world 系的 z=0.75 m；T_world_from_camera 已标定。
result = intersect_image_plane(
    picked_pixels, camera, [0, 0, 1], -0.75,
    T_frame_from_camera=T_world_from_camera,
    distortion=distortion_coefficients,
    min_incidence_cos=0.05,
)
world_points = result['points_frame'][result['valid']]
```

平面偏移、平移、输出点和 `max_ray_distance` 必须使用同一物理单位；输入毫米则输出毫米。平面法向量允许任意非零长度，内部归一化。只接受沿射线前方且位于最大距离内的交点；`min_incidence_cos` 过滤近乎平行的射线，因为这时极小像素误差能造成很大的位置误差。无效交点返回 NaN 和 `valid=False`，不会拿它作为真实深度。

畸变遵循 OpenCV 标准 4/5/8/12/14 参数模型，不支持 fisheye。反畸变先跑 8 轮，再**正向投影验算**；未达 `max_reprojection_error_px` 的射线加跑至 50 轮，仍不满足则判无效。默认误差门限 1e-6 px。传入已经去畸变的像素时应设 `distortion=None`，并使用对应的新内参。像素即便位于图像边界外也可计算射线，但此接口不验证遮挡、台面实际位置或手眼标定；现实抓取前仍需做这些验证。

5 项专项测试包括针孔解析真值、任意刚体相机/世界系变换、五种标准畸变长度、严格逆向精度、近平行/背后/范围限制、输入错误。合成 300 点，0.15 px 高斯像素噪声，世界平面 z=2 m：平均位置误差 0.451 mm，P95 为 0.910 mm；这是特定相机几何和噪声条件，不是现场精度。20,000 像素完整 API 短基准：平面求交无畸变中位 4.88–4.92 ms；含标准反畸变由优化前 22.92 ms 降至 20.15 ms，并保留逐射线反投影核查。两次原始报告在 `projects/output/continuous_batch2/ray_plane_benchmark` 与 `ray_plane_benchmark_optimized`。
